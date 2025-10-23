# openai_wrapper_batch.py
# Simple, importable batch wrapper around the OpenAI Responses API with
# optional conversation history (batched across parallel threads).
#
# New in this version:
# - BatchConversation.ask_masked([...Optional[str]...]): send a user turn only for
#   selected threads (None = skip that thread this step). Returns a
#   list aligned to thread indices with None for skipped threads.
# - BatchConversation.ask_map({idx: prompt}): tiny convenience wrapper over ask_masked.
#
# This lets you drive many conversations in parallel (e.g., 1 per problem),
# even when they have different numbers of steps/actions, without paying for
# completed threads or forcing no-op turns.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Iterable, Union, Tuple
import json
import io
import time
import tempfile
import uuid

from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 16 * 1024
DEFAULT_COMPLETION_WINDOW = "24h"  # Batch window

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# ------------------- Utilities -------------------

def _as_list(x: Union[str, List[str], None], n: int, name: str) -> List[Optional[str]]:
    """
    Broadcast scalars to length n; validate lists have matching length.
    Allows None for optional fields.
    """
    if x is None:
        return [None] * n
    if isinstance(x, list):
        if len(x) != n:
            raise ValueError(f"{name} length {len(x)} must match n={n}")
        return x
    if isinstance(x, str):
        return [x] * n
    raise TypeError(f"{name} must be str | list[str] | None")

def _extract_output_text(body: Dict[str, Any]) -> str:
    """
    Extract plain text from Responses API response body (batch shape).
    """
    if isinstance(body, dict) and body.get("output_text"):
        return str(body["output_text"]).strip()

    pieces: List[str] = []
    for item in body.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                t = c.get("text", "")
                if t:
                    pieces.append(t)
    return "\n".join(pieces).strip()

def _write_jsonl(records: Iterable[Dict[str, Any]]) -> str:
    """
    Write JSONL to a temp file and return its path.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    with tmp as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp.name

def _submit_batch(jsonl_path: str, completion_window: str) -> Tuple[str, str]:
    """
    Upload a JSONL file for /v1/responses batch and create the batch job.
    Returns (batch_id, output_file_id when done).
    """
    client = _get_client()
    in_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return batch.id, None

def _poll_for_completion(batch_id: str, poll_interval: float = 5.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Poll the batch until it completes. Returns the final batch object (dict-like).
    """
    client = _get_client()
    statuses_active = { "validating", "in_progress", "finalizing" }
    last = None
    while True:
        b = client.batches.retrieve(batch_id)
        status = getattr(b, "status", None)
        if verbose and status != last:
            print(f"[batch {batch_id}] status = {status}")
            last = status
        if status not in statuses_active:
            return b
        time.sleep(poll_interval)

def _download_output_file(file_id: str) -> List[Dict[str, Any]]:
    """
    Download the output JSONL (as bytes), parse per-line JSON, and return list of dicts.
    """
    client = _get_client()
    stream = client.files.content(file_id)
    raw = stream.read()
    records = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

def _assemble_jsonl_lines(
    messages_per_item: List[List[Dict[str, str]]],
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    run_tag: Optional[str] = None,
    namespace: str = "job",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build batch input JSONL lines for /v1/responses. Returns (lines, custom_ids).
    """
    lines: List[Dict[str, Any]] = []
    custom_ids: List[str] = []
    run_tag = run_tag or uuid.uuid4().hex[:10]
    for idx, msgs in enumerate(messages_per_item):
        custom_id = f"{namespace}::{run_tag}::{idx}"
        body = {
            "model": model,
            "input": msgs,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": int(max_output_tokens),
        }
        lines.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        })
        custom_ids.append(custom_id)
    return lines, custom_ids

def _run_batch_and_collect(
    lines: List[Dict[str, Any]],
    completion_window: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Submit, poll, and return output records (parsed JSON from output JSONL).
    """
    jsonl_path = _write_jsonl(lines)
    try:
        batch_id, _ = _submit_batch(jsonl_path, completion_window)
        final = _poll_for_completion(batch_id, verbose=verbose)
        if getattr(final, "status", None) != "completed":
            err_id = getattr(final, "error_file_id", None) or getattr(final, "errors_file_id", None)
            if err_id:
                errs = _download_output_file(err_id)
                raise RuntimeError(f"Batch failed with status={final.status}. Errors: {errs[:3]}")
            raise RuntimeError(f"Batch finished with status={final.status}")
        out_id = getattr(final, "output_file_id", None)
        if not out_id:
            raise RuntimeError("Completed batch missing output_file_id")
        return _download_output_file(out_id)
    finally:
        try:
            import os
            os.remove(jsonl_path)
        except Exception:
            pass

# ----------------- Public: basic batch -----------------

def batch_single_prompt(
    prompts: Union[str, List[str]],
    system_prompt: Optional[Union[str, List[str]]] = None,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    run_tag: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Submit one or many prompts as a single batch job.
    Always returns a list of assistant strings in the same order as inputs.
    """
    items = prompts if isinstance(prompts, list) else [prompts]
    n = len(items)
    systems = _as_list(system_prompt, n, "system_prompt")

    messages_per_item: List[List[Dict[str, str]]] = []
    for i in range(n):
        msgs: List[Dict[str, str]] = []
        if systems[i]:
            msgs.append({"role": "system", "content": systems[i]})  # type: ignore
        msgs.append({"role": "user", "content": items[i]})
        messages_per_item.append(msgs)

    lines, custom_ids = _assemble_jsonl_lines(
        messages_per_item=messages_per_item,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        run_tag=run_tag,
        namespace="single",
    )
    out_records = _run_batch_and_collect(lines, completion_window, verbose=verbose)

    outputs: Dict[str, str] = {}
    for r in out_records:
        cid = r.get("custom_id")
        resp = (r.get("response") or {})
        status = resp.get("status_code")
        body = resp.get("body") or {}
        if status != 200:
            raise RuntimeError(f"Non-200 for {cid}: {status}")
        outputs[cid] = _extract_output_text(body)

    return [outputs[cid] for cid in custom_ids]

# ----------------- Public: batched conversation -----------------

@dataclass
class _Turn:
    role: str
    content: str
    ts: float

    def to_msg(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class BatchConversation:
    """
    Maintain N parallel conversation histories and advance them using batch jobs.
    Each call to .ask(...) adds a user turn (broadcast or per-thread list) and
    submits one batch; assistant replies are appended to each thread's history.

    New: .ask_masked([...Optional[str]...]) lets you advance only the threads that
    still have pending turns (None = skip).
    """

    def __init__(
        self,
        count: int = 1,
        system_prompt: Optional[Union[str, List[str]]] = None,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
        run_tag: Optional[str] = None,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be >= 1")
        self.count = count
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.completion_window = completion_window
        self.run_tag = run_tag or uuid.uuid4().hex[:10]
        self.step = 0

        systems = _as_list(system_prompt, count, "system_prompt")
        self._threads: List[List[_Turn]] = []
        now = time.time()
        for i in range(count):
            turns: List[_Turn] = []
            if systems[i]:
                turns.append(_Turn("system", systems[i], now))  # type: ignore
            self._threads.append(turns)

    # ---- persistence ----
    def save(self, path: str) -> None:
        data = {
            "count": self.count,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "completion_window": self.completion_window,
            "run_tag": self.run_tag,
            "step": self.step,
            "threads": [
                [asdict(t) for t in thread] for thread in self._threads
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BatchConversation":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(
            count=int(data["count"]),
            system_prompt=None,
            model=data.get("model", DEFAULT_MODEL),
            reasoning_effort=data.get("reasoning_effort", DEFAULT_EFFORT),
            max_output_tokens=int(data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
            completion_window=data.get("completion_window", DEFAULT_COMPLETION_WINDOW),
            run_tag=data.get("run_tag"),
        )
        obj.step = int(data.get("step", 0))
        obj._threads = [
            [ _Turn(**t) for t in thread ]
            for thread in data.get("threads", [])
        ]
        return obj

    # ---- read-only views ----
    def history(self, idx: int) -> List[Dict[str, str]]:
        return [t.to_msg() for t in self._threads[idx]]

    def histories(self) -> List[List[Dict[str, str]]]:
        return [[t.to_msg() for t in thread] for thread in self._threads]

    # ---- main ops ----
    def ask(self, user_prompts: Union[str, List[str]], *, verbose: bool = True) -> List[str]:
        """
        Add a user turn (broadcast or list) across threads, submit a batch,
        append assistant replies, and return the replies (list[str]).
        """
        prompts = user_prompts if isinstance(user_prompts, list) else [user_prompts] * self.count
        if len(prompts) != self.count:
            raise ValueError(f"user_prompts length {len(prompts)} must match count={self.count}")

        now = time.time()
        for i in range(self.count):
            self._threads[i].append(_Turn("user", prompts[i], now))

        messages_per_item: List[List[Dict[str, str]]] = []
        for i in range(self.count):
            messages_per_item.append([t.to_msg() for t in self._threads[i]])

        ns = f"conv::step{self.step}"
        lines, custom_ids = _assemble_jsonl_lines(
            messages_per_item=messages_per_item,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            run_tag=self.run_tag,
            namespace=ns,
        )
        out_records = _run_batch_and_collect(lines, self.completion_window, verbose=verbose)

        by_id: Dict[str, str] = {}
        for r in out_records:
            cid = r.get("custom_id")
            resp = (r.get("response") or {})
            status = resp.get("status_code")
            body = resp.get("body") or {}
            if status != 200:
                raise RuntimeError(f"Non-200 for {cid}: {status}")
            by_id[cid] = _extract_output_text(body)

        replies: List[str] = []
        for cid in custom_ids:
            replies.append(by_id[cid])

        now2 = time.time()
        for i in range(self.count):
            self._threads[i].append(_Turn("assistant", replies[i], now2))

        self.step += 1
        return replies

    def ask_masked(self, user_prompts_masked: List[Optional[str]], *, verbose: bool = True) -> List[Optional[str]]:
        """
        Add a user turn for *selected* threads only (None = skip that thread),
        submit one batch containing just the active threads, and return a list
        of replies aligned to thread indices (None where skipped).
        """
        if len(user_prompts_masked) != self.count:
            raise ValueError(f"user_prompts_masked length {len(user_prompts_masked)} must match count={self.count}")

        # Determine active threads for this step
        active_indices = [i for i, p in enumerate(user_prompts_masked) if p is not None]
        if not active_indices:
            return [None] * self.count  # no-op step, no increment

        now = time.time()
        for i in active_indices:
            self._threads[i].append(_Turn("user", user_prompts_masked[i] or "", now))

        # Build payload only for active threads
        messages_per_item: List[List[Dict[str, str]]] = []
        id2slot: Dict[str, int] = {}  # custom_id -> thread index
        ns = f"conv::step{self.step}"

        # Assemble lines *in active order*; track mapping to thread index
        for i in active_indices:
            msgs = [t.to_msg() for t in self._threads[i]]
            messages_per_item.append(msgs)

        lines, custom_ids = _assemble_jsonl_lines(
            messages_per_item=messages_per_item,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            run_tag=self.run_tag,
            namespace=ns,
        )
        # Map back each custom_id to its thread index
        for slot, cid in zip(active_indices, custom_ids):
            id2slot[cid] = slot

        out_records = _run_batch_and_collect(lines, self.completion_window, verbose=verbose)

        # Collect outputs
        slot_to_reply: Dict[int, str] = {}
        for r in out_records:
            cid = r.get("custom_id")
            resp = (r.get("response") or {})
            status = resp.get("status_code")
            body = resp.get("body") or {}
            if status != 200:
                raise RuntimeError(f"Non-200 for {cid}: {status}")
            slot = id2slot[cid]
            slot_to_reply[slot] = _extract_output_text(body)

        # Append assistant turns for active threads
        now2 = time.time()
        for i in active_indices:
            reply = slot_to_reply[i]
            self._threads[i].append(_Turn("assistant", reply, now2))

        # Build aligned result (None where skipped)
        out: List[Optional[str]] = [None] * self.count
        for i in active_indices:
            out[i] = slot_to_reply[i]

        self.step += 1
        return out

    def ask_map(self, prompt_map: Dict[int, str], *, verbose: bool = True) -> Dict[int, str]:
        """
        Convenience wrapper: provide {thread_index: prompt}. Returns {thread_index: reply}.
        """
        masked: List[Optional[str]] = [None] * self.count
        for i, p in prompt_map.items():
            if i < 0 or i >= self.count:
                raise IndexError(f"thread index out of range: {i}")
            masked[i] = p
        replies = self.ask_masked(masked, verbose=verbose)
        return {i: (replies[i] or "") for i in prompt_map.keys()}
