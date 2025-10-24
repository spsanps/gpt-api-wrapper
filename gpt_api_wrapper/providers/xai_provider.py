"""xAI Grok provider implementation."""
from __future__ import annotations

import logging
import os
from typing import List, Sequence

from .base import BaseProvider, Message, ProviderNotAvailableError, register_provider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class XAIProvider(OpenAIProvider):
    name = "xai"

    def _create_client(self):  # pragma: no cover - IO heavy
        from openai import OpenAI

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ProviderNotAvailableError("XAI_API_KEY is not set")
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        logger.debug("Initialising XAI client base_url=%s", base_url)
        return OpenAI(api_key=api_key, base_url=base_url)

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
        logger.info("xAI batch using sequential fallback (API has no batch endpoint)")
        return BaseProvider.batch_generate(
            self,
            batched_messages,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            completion_window=completion_window,
            verbose=verbose,
        )


register_provider(XAIProvider())
