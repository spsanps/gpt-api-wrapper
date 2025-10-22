# Simple single prompt wrapper (non-batch) using OpenAI Responses API
# Assumes OPENAI_API_KEY (and optional OPENAI_BASE_URL) are set in environment.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Any, Union
import json
import time
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"  # adjust if you want a different model
DEFAULT_EFFORT = "medium"  # per batch script defaults
DEFAULT_MAX_OUTPUT_TOKENS = 16*1024  # lower than batch max for ad-hoc calls

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def run_single_prompt(prompt: str,
                      model: str = DEFAULT_MODEL,
                      reasoning_effort: str = DEFAULT_EFFORT,
                      max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                      system_prompt: str = None) -> str:
    """Send a single user prompt via Responses API and return plain text output.

    Parameters:
        prompt: The user message string.
        model: Model name (default gpt-5 set in batch script).
        reasoning_effort: One of low|medium|high.
        max_output_tokens: Upper bound on generated tokens.
        system_prompt: Optional system prompt to include.

    Returns:
        Extracted text response.
    """
    client = _get_client()
    
    # Build input messages
    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": prompt})
    
    resp = client.responses.create(
        model=model,
        input=input_messages,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )

    # Prefer SDK convenience if present
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    # Fallback: traverse structured output
    pieces: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", "")
                if t:
                    pieces.append(t)
    return "".join(pieces).strip()


# ---- Conversation primitives ----

@dataclass
class ChatTurn:
    role: str  # "system" | "user" | "assistant"
    content: str
    ts: float = time.time()

    def to_msg(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

class Conversation:
    """
    Lightweight multi-turn conversation manager for the OpenAI Responses API.

    Usage:
        convo = Conversation(system_prompt="You are a concise assistant.")
        reply = convo.ask("Hello there!")
        reply = convo.ask("Summarize that in 5 words.")
        convo.save("my_convo.json")
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_turns: int = 32,  # keep last N user/assistant turns (+ optional system)
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self._turns: List[ChatTurn] = []
        if system_prompt:
            self.set_system(system_prompt)

    # ----- History management -----

    def set_system(self, system_prompt: str) -> None:
        # Replace existing system message (if any)
        self._turns = [t for t in self._turns if t.role != "system"]
        self._turns.insert(0, ChatTurn("system", system_prompt))

    def add_user(self, text: str) -> None:
        self._turns.append(ChatTurn("user", text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._turns.append(ChatTurn("assistant", text))
        self._trim()

    def history(self) -> List[Dict[str, str]]:
        return [t.to_msg() for t in self._turns]

    def reset(self, keep_system: bool = True) -> None:
        if keep_system:
            sys_msgs = [t for t in self._turns if t.role == "system"]
            self._turns = sys_msgs
        else:
            self._turns = []

    def undo_last_turn(self) -> None:
        # Remove last non-system message (useful after an unwanted reply)
        for i in range(len(self._turns) - 1, -1, -1):
            if self._turns[i].role != "system":
                del self._turns[i]
                break

    def _trim(self) -> None:
        """
        Keep only the most recent `max_turns` non-system messages,
        plus a single system message at the front if present.
        """
        sys_msgs = [t for t in self._turns if t.role == "system"]
        chat_msgs = [t for t in self._turns if t.role != "system"]
        if len(chat_msgs) > self.max_turns:
            chat_msgs = chat_msgs[-self.max_turns :]
        self._turns = (sys_msgs[:1] + chat_msgs) if sys_msgs else chat_msgs

    # ----- I/O (persist/restore) -----

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self._turns], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Conversation":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        convo = cls(**kwargs)
        convo._turns = [ChatTurn(**t) for t in raw]
        return convo

    # ----- Calling the Responses API -----

    def ask(
        self,
        user_text: str,
        *,
        system_override: Optional[str] = None,
        stream: bool = False,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Union[str, Iterable[str]]:
        """
        Add a user turn, call the model, store the assistant reply, and return it.

        If stream=True, returns an iterator of string chunks; the final assembled
        text is added to history automatically once the stream completes.
        """
        # Build message list
        if system_override is not None:
            base_msgs = [{"role": "system", "content": system_override}]
            base_msgs += [t.to_msg() for t in self._turns if t.role != "system"]
        else:
            base_msgs = self.history()

        base_msgs.append({"role": "user", "content": user_text})
        if extra_messages:
            base_msgs.extend(extra_messages)

        self.add_user(user_text)

        use_model = model or self.model
        use_effort = reasoning_effort or self.reasoning_effort
        use_max = max_output_tokens or self.max_output_tokens

        client = _get_client()

        if stream:
            # Streaming via Responses API
            # Collect text chunks as they arrive, then commit to history.
            from contextlib import contextmanager

            @contextmanager
            def _streaming():
                s = client.responses.stream(
                    model=use_model,
                    input=base_msgs,
                    reasoning={"effort": use_effort},
                    max_output_tokens=use_max,
                )
                try:
                    yield s
                finally:
                    s.close()

            def _generator():
                pieces = []
                with _streaming() as s:
                    for event in s:
                        if event.type == "response.output_text.delta":
                            chunk = event.delta
                            if chunk:
                                pieces.append(chunk)
                                yield chunk
                    # finalize and add to history
                    s.finalize()
                    full = "".join(pieces).strip()
                    self.add_assistant(full)
            return _generator()

        # Non-streaming single-shot
        resp = client.responses.create(
            model=use_model,
            input=base_msgs,
            reasoning={"effort": use_effort},
            max_output_tokens=use_max,
        )

        if hasattr(resp, "output_text") and resp.output_text:
            text = resp.output_text.strip()
        else:
            # fallback traversal (same as your run_single_prompt)
            chunks: list[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        t = getattr(c, "text", "")
                        if t:
                            chunks.append(t)
            text = "".join(chunks).strip()

        self.add_assistant(text)
        return text

# ---- Tiny convenience wrapper if you like a functional style ----

def run_conversation_turn(
    convo: Conversation,
    user_text: str,
    **kwargs,
) -> str:
    return convo.ask(user_text, **kwargs) if isinstance(convo, Conversation) else str(run_single_prompt(user_text))
