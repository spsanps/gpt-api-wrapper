"""Provider registry and abstractions for GPT API wrapper."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

Message = Dict[str, str]


class ProviderNotAvailableError(RuntimeError):
    """Raised when a provider cannot be initialised due to missing dependency."""


class BaseProvider:
    """Base interface for provider backends."""

    name: str = "base"
    supports_streaming: bool = False

    def __init__(self) -> None:
        self._client = None

    def ensure_client(self):  # pragma: no cover - simple getter
        if self._client is None:
            self._client = self._create_client()
        return self._client

    # -- hooks -----------------------------------------------------------------
    def _create_client(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        stream: bool = False,
    ) -> str | Iterable[str]:  # pragma: no cover - abstract
        raise NotImplementedError

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
        """Default batch implementation iterating sequentially."""

        logger.debug("Falling back to sequential batch execution for %s", self.name)
        results: List[str] = []
        if not batched_messages:
            return results

        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception:  # pragma: no cover - optional dep
            def _iter(x: Iterable[int]) -> Iterator[int]:
                return iter(x)
            iterator = _iter(range(len(batched_messages)))
        else:  # pragma: no cover - UI component, hard to test
            iterator = tqdm(
                range(len(batched_messages)),
                desc=f"{self.name} batch",
                disable=not verbose,
            )

        for idx in iterator:
            messages = batched_messages[idx]
            logger.debug("Running fallback batch item %s", idx)
            result = self.generate(
                messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                stream=False,
            )
            assert isinstance(result, str), "Non-stream results expected for batch fallback"
            results.append(result)
        if hasattr(iterator, "close"):
            iterator.close()
        return results


_registry: Dict[str, BaseProvider] = {}


def register_provider(provider: BaseProvider) -> None:
    key = provider.name.lower()
    if key in _registry:
        logger.warning("Provider %s already registered, overriding", key)
    _registry[key] = provider


def get_provider(name: Optional[str]) -> BaseProvider:
    key = (name or "openai").lower()
    if key not in _registry:
        raise KeyError(f"Unknown provider: {name}")
    return _registry[key]


def available_providers() -> List[str]:
    return sorted(_registry.keys())
