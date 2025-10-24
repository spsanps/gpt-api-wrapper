"""Batch helpers delegating to provider backends."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Union

from gpt_api_wrapper.openai_wrapper import DEFAULT_EFFORT, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER
from gpt_api_wrapper.providers import get_provider

logger = logging.getLogger(__name__)

DEFAULT_COMPLETION_WINDOW = "24h"

Message = Dict[str, str]


def _broadcast_list(
    value: Optional[Union[str, Sequence[Optional[str]]]],
    n: int,
    *,
    name: str,
) -> List[Optional[str]]:
    if value is None:
        return [None] * n
    if isinstance(value, str):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} length {len(value)} must match n={n}")
    return list(value)


def _messages_from_prompt(prompt: str, system_prompt: Optional[str]) -> List[Message]:
    messages: List[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def batch_single_prompt(
    prompts: Union[str, Sequence[str]],
    system_prompt: Optional[Union[str, Sequence[Optional[str]]]] = None,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    run_tag: Optional[str] = None,
    verbose: bool = True,
    provider: str = DEFAULT_PROVIDER,
) -> List[str]:
    """Submit one or many prompts as a batch job using the requested provider."""

    if isinstance(prompts, str):
        items = [prompts]
    else:
        items = list(prompts)
    n = len(items)
    systems = _broadcast_list(system_prompt, n, name="system_prompt")
    backend = get_provider(provider)

    messages_per_item = [
        _messages_from_prompt(items[i], systems[i]) for i in range(n)
    ]
    if run_tag:
        logger.info("Running batch_single_prompt via %s (items=%d, tag=%s)", provider, n, run_tag)
    else:
        logger.info("Running batch_single_prompt via %s (items=%d)", provider, n)
    outputs = backend.batch_generate(
        messages_per_item,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        completion_window=completion_window,
        verbose=verbose,
    )
    return outputs


@dataclass
class _Turn:
    role: str
    content: str
    ts: float

    def to_msg(self) -> Message:
        return {"role": self.role, "content": self.content}


class BatchConversation:
    """Maintain parallel conversation threads driven via provider batch calls."""

    def __init__(
        self,
        count: int = 1,
        system_prompt: Optional[Union[str, Sequence[Optional[str]]]] = None,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be >= 1")
        self.count = count
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.completion_window = completion_window
        self.provider_name = provider
        self._provider = get_provider(provider)
        now = time.time()
        systems = _broadcast_list(system_prompt, count, name="system_prompt")
        self._threads: List[List[_Turn]] = []
        for i in range(count):
            turns: List[_Turn] = []
            if systems[i]:
                turns.append(_Turn("system", systems[i] or "", now))
            self._threads.append(turns)
        self.step = 0

    def save(self, path: str) -> None:
        data = {
            "count": self.count,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "completion_window": self.completion_window,
            "provider": self.provider_name,
            "step": self.step,
            "threads": [
                [asdict(t) for t in thread] for thread in self._threads
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BatchConversation":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        obj = cls(
            count=int(data.get("count", 1)),
            system_prompt=None,
            model=data.get("model", DEFAULT_MODEL),
            reasoning_effort=data.get("reasoning_effort", DEFAULT_EFFORT),
            max_output_tokens=int(data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
            completion_window=data.get("completion_window", DEFAULT_COMPLETION_WINDOW),
            provider=data.get("provider", DEFAULT_PROVIDER),
        )
        obj.step = int(data.get("step", 0))
        obj._threads = [
            [_Turn(**turn) for turn in thread]
            for thread in data.get("threads", [])
        ]
        return obj

    def history(self, idx: int) -> List[Message]:
        return [t.to_msg() for t in self._threads[idx]]

    def histories(self) -> List[List[Message]]:
        return [[t.to_msg() for t in thread] for thread in self._threads]

    def _run_batch(self, payload: List[List[Message]], verbose: bool) -> List[str]:
        return self._provider.batch_generate(
            payload,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            completion_window=self.completion_window,
            verbose=verbose,
        )

    def ask(self, user_prompts: Union[str, Sequence[str]], *, verbose: bool = True) -> List[str]:
        if isinstance(user_prompts, str):
            prompts = [user_prompts] * self.count
        else:
            prompts = list(user_prompts)
        if len(prompts) != self.count:
            raise ValueError(f"user_prompts length {len(prompts)} must match count={self.count}")

        now = time.time()
        for i in range(self.count):
            self._threads[i].append(_Turn("user", prompts[i], now))

        payload: List[List[Message]] = []
        for i in range(self.count):
            payload.append([t.to_msg() for t in self._threads[i]])

        logger.info("BatchConversation step=%d via %s", self.step, self.provider_name)
        replies = self._run_batch(payload, verbose)

        now2 = time.time()
        for i in range(self.count):
            self._threads[i].append(_Turn("assistant", replies[i], now2))

        self.step += 1
        return replies

    def ask_masked(self, user_prompts_masked: Sequence[Optional[str]], *, verbose: bool = True) -> List[Optional[str]]:
        if len(user_prompts_masked) != self.count:
            raise ValueError("user_prompts_masked length must match count")

        active_indices = [i for i, prompt in enumerate(user_prompts_masked) if prompt is not None]
        if not active_indices:
            return [None] * self.count

        now = time.time()
        for i in active_indices:
            self._threads[i].append(_Turn("user", user_prompts_masked[i] or "", now))

        payload: List[List[Message]] = []
        slot_map: Dict[int, int] = {}
        for slot, idx in enumerate(active_indices):
            payload.append([t.to_msg() for t in self._threads[idx]])
            slot_map[slot] = idx

        logger.info(
            "BatchConversation masked step=%d via %s active=%s",
            self.step,
            self.provider_name,
            active_indices,
        )
        replies = self._run_batch(payload, verbose)

        result: List[Optional[str]] = [None] * self.count
        now2 = time.time()
        for slot, reply in enumerate(replies):
            idx = slot_map[slot]
            self._threads[idx].append(_Turn("assistant", reply, now2))
            result[idx] = reply

        self.step += 1
        return result

    def ask_map(self, prompt_map: Dict[int, str], *, verbose: bool = True) -> Dict[int, str]:
        masked: List[Optional[str]] = [None] * self.count
        for idx, prompt in prompt_map.items():
            if idx < 0 or idx >= self.count:
                raise IndexError(f"thread index out of range: {idx}")
            masked[idx] = prompt
        replies = self.ask_masked(masked, verbose=verbose)
        return {idx: replies[idx] or "" for idx in prompt_map.keys()}
