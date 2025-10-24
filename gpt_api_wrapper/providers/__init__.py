"""Provider registry for GPT API wrapper."""
from __future__ import annotations

from .base import BaseProvider, ProviderNotAvailableError, available_providers, get_provider, register_provider

# Import concrete providers to trigger registration side effects
from . import openai_provider  # noqa: F401
from . import anthropic_provider  # noqa: F401
from . import xai_provider  # noqa: F401
from . import gemini_provider  # noqa: F401

__all__ = [
    "BaseProvider",
    "ProviderNotAvailableError",
    "available_providers",
    "get_provider",
    "register_provider",
]
