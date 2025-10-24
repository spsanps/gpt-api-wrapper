"""Generic conversation utilities built on top of provider backends."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Iterable
from typing import Dict, List, Optional, Sequence, Union

from gpt_api_wrapper.providers import get_provider

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 16 * 1024
DEFAULT_PROVIDER = "openai"

logger = logging.getLogger(__name__)

Message = Dict[str, str]


def _prepare_messages(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    extra: Optional[Sequence[Message]] = None,
) -> List[Message]:
    messages: List[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if extra:
        messages.extend(extra)
    return messages
def run_single_prompt(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    system_prompt: Optional[str] = None,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """Send a single user prompt to the configured provider and return text."""

    backend = get_provider(provider)
    messages = _prepare_messages(prompt, system_prompt=system_prompt)
    logger.info("Running single prompt via %s", provider)
    result = backend.generate(
        messages,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        stream=False,
    )
    if isinstance(result, str):
        return result.strip()
    text = "".join(result).strip()
    return text


@dataclass
class ChatTurn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)

    def to_msg(self) -> Message:
        return {"role": self.role, "content": self.content}


class Conversation:
    """Multi-turn conversation manager supporting multiple providers."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_turns: int = 32,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.provider_name = provider
        self._turns: List[ChatTurn] = []
        if system_prompt:
            self.set_system(system_prompt)

    # ----- History management -----

    def set_system(self, system_prompt: str) -> None:
        self._turns = [t for t in self._turns if t.role != "system"]
        self._turns.insert(0, ChatTurn("system", system_prompt))

    def add_user(self, text: str) -> None:
        self._turns.append(ChatTurn("user", text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._turns.append(ChatTurn("assistant", text))
        self._trim()

    def history(self) -> List[Message]:
        return [t.to_msg() for t in self._turns]

    def reset(self, keep_system: bool = True) -> None:
        if keep_system:
            systems = [t for t in self._turns if t.role == "system"]
            self._turns = systems
        else:
            self._turns = []

    def undo_last_turn(self) -> None:
        for i in range(len(self._turns) - 1, -1, -1):
            if self._turns[i].role != "system":
                del self._turns[i]
                break

    def _trim(self) -> None:
        systems = [t for t in self._turns if t.role == "system"]
        chats = [t for t in self._turns if t.role != "system"]
        if len(chats) > self.max_turns:
            chats = chats[-self.max_turns :]
        self._turns = (systems[:1] + chats) if systems else chats

    # ----- Persistence -----

    def save(self, path: str) -> None:
        data = {
            "provider": self.provider_name,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "max_turns": self.max_turns,
            "turns": [asdict(t) for t in self._turns],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Conversation":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        params = {
            "provider": data.get("provider", kwargs.pop("provider", DEFAULT_PROVIDER)),
            "model": data.get("model", kwargs.pop("model", DEFAULT_MODEL)),
            "reasoning_effort": data.get("reasoning_effort", kwargs.pop("reasoning_effort", DEFAULT_EFFORT)),
            "max_output_tokens": int(data.get("max_output_tokens", kwargs.pop("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))),
            "max_turns": int(data.get("max_turns", kwargs.pop("max_turns", 32))),
        }
        params.update(kwargs)
        convo = cls(**params)
        convo._turns = [ChatTurn(**turn) for turn in data.get("turns", [])]
        return convo

    # ----- Main operation -----

    def ask(
        self,
        user_text: str,
        *,
        system_override: Optional[str] = None,
        stream: bool = False,
        extra_messages: Optional[List[Message]] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> Union[str, Iterable[str]]:
        base_msgs = []
        if system_override is not None:
            base_msgs.append({"role": "system", "content": system_override})
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
        provider_name = provider or self.provider_name
        backend = get_provider(provider_name)

        if stream:
            if not backend.supports_streaming:
                raise NotImplementedError(f"Provider {provider_name} does not support streaming")
            logger.info("Streaming conversation turn via %s", provider_name)
            iterator = backend.generate(
                base_msgs,
                model=use_model,
                reasoning_effort=use_effort,
                max_output_tokens=use_max,
                stream=True,
            )
            if not isinstance(iterator, Iterable):
                raise RuntimeError("Expected iterable from streaming provider")

            def _gen() -> Iterable[str]:
                pieces: List[str] = []
                for chunk in iterator:
                    pieces.append(chunk)
                    yield chunk
                full = "".join(pieces).strip()
                self.add_assistant(full)

            return _gen()

        logger.info("Running conversation turn via %s", provider_name)
        result = backend.generate(
            base_msgs,
            model=use_model,
            reasoning_effort=use_effort,
            max_output_tokens=use_max,
            stream=False,
        )
        if isinstance(result, str):
            text = result.strip()
        else:
            text = "".join(result).strip()
        self.add_assistant(text)
        return text


def run_conversation_turn(convo: Conversation, user_text: str, **kwargs) -> str:
    return convo.ask(user_text, **kwargs) if isinstance(convo, Conversation) else str(run_single_prompt(user_text, **kwargs))
