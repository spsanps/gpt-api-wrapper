"""Anthropic Claude provider implementation."""
from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Sequence, Tuple

from .base import BaseProvider, Message, ProviderNotAvailableError, register_provider

logger = logging.getLogger(__name__)


def _split_system_message(messages: Sequence[Message]) -> Tuple[str | None, List[Message]]:
    system = None
    converted: List[Message] = []
    for msg in messages:
        if msg.get("role") == "system" and system is None:
            system = msg.get("content")
            continue
        converted.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    return system, converted


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    supports_streaming = True

    def _create_client(self):  # pragma: no cover - IO heavy
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ProviderNotAvailableError("anthropic package is not installed") from exc

        return anthropic.Anthropic()

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        stream: bool = False,
    ) -> str | Iterable[str]:
        client = self.ensure_client()
        system, rest = _split_system_message(messages)
        request_kwargs = {
            "model": model,
            "messages": rest,
            "max_tokens": max_output_tokens,
        }
        if system:
            request_kwargs["system"] = system

        if reasoning_effort and reasoning_effort != "medium":
            logger.debug("Anthropic provider ignoring reasoning_effort=%s", reasoning_effort)

        if stream:
            def _generator() -> Iterator[str]:
                with client.messages.stream(**request_kwargs) as stream_obj:  # type: ignore[arg-type]
                    for event in stream_obj:
                        delta = getattr(event, "delta", None)
                        if delta is not None:
                            text = getattr(delta, "text", "")
                            if text:
                                yield text
                    stream_obj.finalize()
            return _generator()

        response = client.messages.create(**request_kwargs)  # type: ignore[arg-type]
        pieces: List[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    pieces.append(text)
        result = "".join(pieces).strip()
        logger.debug("Anthropic response length=%d", len(result))
        return result

    def batch_generate(
        self,
        batched_messages: Sequence[Sequence[Message]],
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        completion_window: str,
        verbose: bool = True,
    ) -> List[str]:
        # Claude batch API is more involved; fall back to sequential for now.
        logger.info("Anthropic batch falling back to sequential processing")
        return super().batch_generate(
            batched_messages,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            completion_window=completion_window,
            verbose=verbose,
        )


register_provider(AnthropicProvider())
