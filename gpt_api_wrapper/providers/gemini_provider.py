"""Google Gemini provider implementation."""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Sequence

from .base import BaseProvider, Message, ProviderNotAvailableError, register_provider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_streaming = False

    def _create_client(self):  # pragma: no cover - IO heavy
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except ImportError as exc:  # pragma: no cover - import guard
            raise ProviderNotAvailableError("google-genai package is not installed") from exc

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderNotAvailableError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")

        genai.configure(api_key=api_key)
        logger.debug("Initialising Gemini client with key prefix=%s", api_key[:4])
        return genai.Client(http_options=HttpOptions(api_version="v1"))

    def _convert_messages(self, messages: Sequence[Message]):
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system" and system_instruction is None:
                system_instruction = text
                continue
            parts = [{"text": text}]
            mapped_role = "user" if role == "user" else "model"
            contents.append({"role": mapped_role, "parts": parts})
        return system_instruction, contents

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        stream: bool = False,
    ) -> str | Iterable[str]:
        if stream:
            raise NotImplementedError("Gemini provider does not yet support streaming")

        client = self.ensure_client()
        system_instruction, contents = self._convert_messages(messages)
        if reasoning_effort and reasoning_effort != "medium":
            logger.debug("Gemini provider ignoring reasoning_effort=%s", reasoning_effort)

        generation_config = {"max_output_tokens": max_output_tokens}

        response = client.models.generate_content(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )
        text = getattr(response, "text", "") or ""
        if not text and hasattr(response, "candidates"):
            for cand in getattr(response, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", [])
                for part in parts or []:
                    piece = getattr(part, "text", None) or getattr(part, "data", "")
                    if piece:
                        text += piece
        text = text.strip()
        logger.debug("Gemini response length=%d", len(text))
        return text

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
        logger.info("Gemini batch using sequential fallback")
        return super().batch_generate(
            batched_messages,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            completion_window=completion_window,
            verbose=verbose,
        )


register_provider(GeminiProvider())
