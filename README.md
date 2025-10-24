# GPT API Wrapper

A lightweight Python library providing simple wrappers around multiple LLM providers (OpenAI, Anthropic Claude, xAI Grok, and Google Gemini). This package supports both single-shot prompts and batched operations with conversation history management.

## Features

- **Single-shot prompts**: Send individual prompts and get responses
- **Conversation management**: Maintain multi-turn conversations with automatic history tracking
- **Batch operations**: Process multiple prompts efficiently using OpenAI's Batch API or provider fallbacks
- **Batched conversations**: Run parallel conversation threads in batch mode across all providers
- **Streaming support**: Stream responses in real-time for single conversations
- **Persistence**: Save and load conversation histories

## Installation

### Install in editable mode (for development)

```bash
pip install -e .
```

### Install from GitHub

You can install directly from the GitHub repository:

**SSH (recommended):**
```bash
pip install "git+ssh://git@github.com/spsanps/gpt-api-wrapper.git@main"
```

**HTTPS:**
```bash
pip install "git+https://github.com/spsanps/gpt-api-wrapper.git@main"
```

**Pin to a specific tag or commit:**
```bash
pip install "git+ssh://git@github.com/spsanps/gpt-api-wrapper.git@<tag-or-commit>"
```

### Requirements

- Python 3.9 or newer
- `openai`, `anthropic`, `xai-sdk`, `google-genai`, and `tqdm` packages (automatically installed)

### Environment Setup

Configure the credentials for the providers you intend to use. Each provider reads its API key from an environment variable.

#### OpenAI

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://custom-endpoint.example.com"  # optional
```

#### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### xAI Grok

```bash
export XAI_API_KEY="your-api-key-here"
export XAI_BASE_URL="https://api.x.ai/v1"  # optional override
```

#### Google Gemini

```bash
export GOOGLE_API_KEY="your-api-key-here"  # or GEMINI_API_KEY
```

## Usage

### Single-shot Prompts

Send a single prompt and get a response:

```python
from gpt_api_wrapper import run_single_prompt, available_providers

response = run_single_prompt(
    "Explain transformers in 2 lines",
    model="gpt-5-mini",
    reasoning_effort="medium",
    system_prompt="You are a concise assistant.",
    provider="openai",  # or "anthropic", "xai", "gemini"
)
print(response)

print("Providers:", available_providers())
```

### Conversation Management

Maintain a multi-turn conversation with automatic history tracking:

```python
from gpt_api_wrapper import Conversation

# Create a conversation with a system prompt
convo = Conversation(
    system_prompt="You are a helpful coding assistant.",
    model="gpt-5-mini",
    reasoning_effort="medium",
    max_turns=32,  # Keep last 32 turns in history
    provider="anthropic"
)

# Ask questions
reply1 = convo.ask("What is a hash map?")
print(reply1)

reply2 = convo.ask("How do I implement one in Python?")
print(reply2)

# Save conversation history
convo.save("my_conversation.json")

# Load and continue later
convo2 = Conversation.load("my_conversation.json")
reply3 = convo2.ask("Can you show me an example?")
```

### Streaming Responses

Stream responses in real-time:

```python
from gpt_api_wrapper import Conversation

convo = Conversation(system_prompt="You are a helpful assistant.")  # streaming currently supported for OpenAI and Anthropic

# Stream the response
for chunk in convo.ask("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
print()  # New line after streaming
```

### Batch Operations

Process multiple prompts efficiently using the Batch API:

```python
from gpt_api_wrapper import batch_single_prompt

prompts = [
    "Explain transformers in 2 lines",
    "Summarize RLHF in 2 lines",
    "What is attention mechanism?"
]

# Process all prompts in a single batch
responses = batch_single_prompt(
    prompts=prompts,
    system_prompt="You are a concise assistant.",
    model="gpt-5-mini",
    reasoning_effort="medium",
    provider="openai",
)

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}\n")
```

### Batched Conversations

Run multiple parallel conversation threads:

```python
from gpt_api_wrapper import BatchConversation

# Create 3 parallel conversation threads
batch_convo = BatchConversation(
    count=3,
    system_prompt="You are a helpful assistant.",
    model="gpt-5-mini",
    provider="gemini"
)

# First turn - different prompts for each thread
replies = batch_convo.ask([
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
])

for i, reply in enumerate(replies):
    print(f"Thread {i+1}: {reply}\n")

# Second turn - broadcast same question to all threads
replies2 = batch_convo.ask("What are its main use cases?")

# Save all conversation histories
batch_convo.save("batch_conversations.json")

# Load and continue later
batch_convo2 = BatchConversation.load("batch_conversations.json")
```

## API Reference

### Single Operations

#### `run_single_prompt(prompt, model, reasoning_effort, max_output_tokens, system_prompt, provider)`

Send a single prompt and return the response.

**Parameters:**
- `prompt` (str): The user message
- `model` (str): Model name (default: "gpt-5-mini")
- `reasoning_effort` (str): One of "low", "medium", "high" (default: "medium")
- `max_output_tokens` (int): Maximum tokens in response (default: 16384)
- `system_prompt` (str, optional): System prompt
- `provider` (str): Provider name (default: "openai")

**Returns:** str - The model's response

#### `Conversation` Class

Multi-turn conversation manager.

**Constructor Parameters:**
- `system_prompt` (str, optional): System prompt for the conversation
- `model` (str): Model name (default: "gpt-5-mini")
- `reasoning_effort` (str): Reasoning effort level (default: "medium")
- `max_output_tokens` (int): Maximum tokens per response (default: 16384)
- `max_turns` (int): Maximum conversation turns to keep in history (default: 32)
- `provider` (str): Provider name (default: "openai")

**Methods:**
- `ask(user_text, stream=False, **kwargs)`: Send a message and get a response
- `save(path)`: Save conversation history to JSON file
- `load(path)`: Load conversation history from JSON file (class method)
- `reset(keep_system=True)`: Clear conversation history
- `set_system(system_prompt)`: Update system prompt
- `history()`: Get conversation history as list of message dicts

### Batch Operations

#### `batch_single_prompt(prompts, system_prompt, model, reasoning_effort, max_output_tokens, completion_window, verbose, provider)`

Process multiple prompts in a single batch job.

For providers without a native batch API (e.g., Anthropic, xAI, Gemini), the wrapper transparently falls back to sequential execution with progress reporting via `tqdm`.

**Parameters:**
- `prompts` (str | list[str]): Single prompt or list of prompts
- `system_prompt` (str | list[str], optional): System prompt(s)
- `model` (str): Model name (default: "gpt-5-mini")
- `reasoning_effort` (str): Reasoning effort (default: "medium")
- `max_output_tokens` (int): Maximum tokens per response (default: 16384)
- `completion_window` (str): Batch completion window (default: "24h")
- `verbose` (bool): Print batch status updates (default: True)
- `provider` (str): Provider name (default: "openai")

**Returns:** list[str] - Responses in the same order as inputs

#### `BatchConversation` Class

Maintain multiple parallel conversation threads processed in batch mode.

**Constructor Parameters:**
- `count` (int): Number of parallel conversation threads
- `system_prompt` (str | list[str], optional): System prompt(s)
- `model` (str): Model name (default: "gpt-5-mini")
- `reasoning_effort` (str): Reasoning effort (default: "medium")
- `max_output_tokens` (int): Maximum tokens per response (default: 16384)
- `completion_window` (str): Batch completion window (default: "24h")
- `provider` (str): Provider name (default: "openai")

**Methods:**
- `ask(user_prompts, verbose=True)`: Send message(s) to all threads and get responses
- `save(path)`: Save all conversation histories to JSON file
- `load(path)`: Load conversation histories from JSON file (class method)
- `history(idx)`: Get history for a specific thread
- `histories()`: Get histories for all threads

## Advanced Usage

### Custom Parameters Per Request

Override default parameters for individual requests:

```python
convo = Conversation(model="gpt-5-mini")

# Use a different model for this request only
response = convo.ask(
    "Explain quantum computing",
    model="gpt-5",
    reasoning_effort="high",
    max_output_tokens=32000
)
```

### System Prompt Override

Temporarily override the system prompt:

```python
convo = Conversation(system_prompt="You are a helpful assistant.")

# Override system prompt for this request only
response = convo.ask(
    "Write a poem",
    system_override="You are a creative poet."
)
```

### Conversation History Management

```python
convo = Conversation(max_turns=10)  # Keep only last 10 turns

# View conversation history
messages = convo.history()
print(messages)

# Reset conversation but keep system prompt
convo.reset(keep_system=True)

# Undo last turn (removes last user-assistant pair)
convo.undo_last_turn()
```

## Testing

Quick sanity check after installation:

```bash
python -c "from gpt_api_wrapper import run_single_prompt, Conversation, batch_single_prompt; print('ok')"
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

[https://github.com/spsanps/gpt-api-wrapper](https://github.com/spsanps/gpt-api-wrapper)
