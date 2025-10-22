"""
GPT API Wrapper - Simple wrapper around OpenAI Responses API

This package provides utilities for interacting with the OpenAI Responses API,
supporting both single-shot prompts and batched operations with conversation history.
"""

from gpt_api_wrapper.openai_wrapper import (
    run_single_prompt,
    Conversation,
    ChatTurn,
    run_conversation_turn,
    DEFAULT_MODEL,
    DEFAULT_EFFORT,
    DEFAULT_MAX_OUTPUT_TOKENS,
)

from gpt_api_wrapper.openai_wrapper_batch import (
    batch_single_prompt,
    BatchConversation,
    DEFAULT_COMPLETION_WINDOW,
)

__all__ = [
    # Single-shot operations
    "run_single_prompt",
    "Conversation",
    "ChatTurn",
    "run_conversation_turn",

    # Batch operations
    "batch_single_prompt",
    "BatchConversation",

    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_COMPLETION_WINDOW",
]

__version__ = "0.1.0"
