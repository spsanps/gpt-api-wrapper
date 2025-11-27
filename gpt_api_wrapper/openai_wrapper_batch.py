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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _broadcast_counts(num_outputs: Union[int, List[int]], n: int, name: str) -> List[int]:
    """Normalize number-of-variant requests to a list of length ``n``."""

    if isinstance(num_outputs, int):
        if num_outputs <= 0:
            raise ValueError(f"{name} must be >= 1")
        return [int(num_outputs)] * n

    if isinstance(num_outputs, list):
        if len(num_outputs) != n:
            raise ValueError(f"{name} length {len(num_outputs)} must match n={n}")
        counts: List[int] = []
        for idx, value in enumerate(num_outputs):
            if not isinstance(value, int):
                raise TypeError(f"{name}[{idx}] must be int")
            if value <= 0:
                raise ValueError(f"{name}[{idx}] must be >= 1")
            counts.append(int(value))
        return counts

    raise TypeError(f"{name} must be int | list[int]")


def _normalize_seed_grid(
    seeds: Optional[Union[int, List[Optional[int]], List[List[Optional[int]]]]],
    counts: List[int],
    name: str,
) -> List[List[Optional[int]]]:
    """
    Expand various seed specifications into a grid parallel to ``counts``.
    Each inner list corresponds to one item and must align with its variant count.
    """

    if seeds is None:
        return [[None] * c for c in counts]

    # Single seed broadcast everywhere
    if isinstance(seeds, int):
        return [[int(seeds)] * c for c in counts]

    if isinstance(seeds, list):
        if not seeds:
            if any(c != 0 for c in counts):
                raise ValueError(f"{name} cannot be an empty list when completions are requested")
            return [[] for _ in counts]

        if all(isinstance(s, list) for s in seeds):
            if len(seeds) != len(counts):
                raise ValueError(f"{name} length {len(seeds)} must match items={len(counts)}")
            grid: List[List[Optional[int]]] = []
            for idx, (sub, count) in enumerate(zip(seeds, counts)):
                if len(sub) != count:
                    raise ValueError(
                        f"{name}[{idx}] length {len(sub)} must match num_outputs={count}"
                    )
                grid.append([s if s is None else int(s) for s in sub])
            return grid

        # Flat list -> broadcast per item; duplicates seeds for each variant of that item
        if len(seeds) != len(counts):
            raise ValueError(f"{name} length {len(seeds)} must match items={len(counts)}")
        grid = []
        for seed, count in zip(seeds, counts):
            grid.append([seed if seed is None else int(seed)] * count)
        return grid

    raise TypeError(f"{name} must be int | list[int|None] | list[list[int|None]] | None")

def _extract_output_text(body: Dict[str, Any]) -> str:
    """
    Extract plain text from Responses API response body (batch shape).
    Handles both regular messages and file_search_call outputs.
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
        elif item.get("type") == "file_search_call":
            # Include file search info if desired
            status = item.get("status", "unknown")
            queries = item.get("queries", [])
            if status == "completed" and queries:
                pieces.append(f"[File search executed: {', '.join(queries)}]")
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
    outputs_per_item: Optional[List[int]] = None,
    seeds_per_item: Optional[List[List[Optional[int]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    include: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[Tuple[int, int]]]:
    """
    Build batch input JSONL lines for /v1/responses.

    Returns a tuple ``(lines, custom_ids, slot_map)`` where slot_map records
    ``(item_index, variant_index)`` for each generated custom_id.
    """

    total_items = len(messages_per_item)
    counts = outputs_per_item or [1] * total_items
    if len(counts) != total_items:
        raise ValueError("outputs_per_item length must match messages_per_item")

    seeds_grid = seeds_per_item or [[None] * c for c in counts]
    if len(seeds_grid) != total_items:
        raise ValueError("seeds_per_item length must match messages_per_item")

    for idx, (seed_row, count) in enumerate(zip(seeds_grid, counts)):
        if len(seed_row) != count:
            raise ValueError(
                f"seeds_per_item[{idx}] length {len(seed_row)} must match outputs_per_item={count}"
            )

    lines: List[Dict[str, Any]] = []
    custom_ids: List[str] = []
    slot_map: List[Tuple[int, int]] = []
    run_tag = run_tag or uuid.uuid4().hex[:10]

    for idx, msgs in enumerate(messages_per_item):
        count = counts[idx]
        seed_row = seeds_grid[idx]
        for variant_idx in range(count):
            suffix = f"::v{variant_idx}" if count > 1 else ""
            custom_id = f"{namespace}::{run_tag}::{idx}{suffix}"
            body: Dict[str, Any] = {
                "model": model,
                "input": msgs,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": int(max_output_tokens),
            }
            seed_value = seed_row[variant_idx]
            if seed_value is not None:
                body["seed"] = int(seed_value)
            if tools is not None:
                body["tools"] = tools
            if include is not None:
                body["include"] = include
            lines.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            })
            custom_ids.append(custom_id)
            slot_map.append((idx, variant_idx))
    return lines, custom_ids, slot_map

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

def recover_batch_result(
    batch_id: str,
    *,
    poll_interval: float = 5.0,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Recover the output from a batch job by its ID.
    
    Polls until the batch completes (if still running) and returns a dictionary
    mapping custom_id to extracted text response. Useful if you lost connection
    or need to retrieve results later.
    
    Parameters
    ----------
    batch_id : str
        The batch job ID to recover
    poll_interval : float
        Seconds between status checks (default: 5.0)
    verbose : bool
        Print status updates (default: True)
        
    Returns
    -------
    dict[str, str]
        Dictionary mapping custom_id to extracted text response
        
    Raises
    ------
    RuntimeError
        If the batch failed or output file is missing
    """
    final = _poll_for_completion(batch_id, poll_interval=poll_interval, verbose=verbose)
    
    if getattr(final, "status", None) != "completed":
        err_id = getattr(final, "error_file_id", None) or getattr(final, "errors_file_id", None)
        if err_id:
            errs = _download_output_file(err_id)
            raise RuntimeError(f"Batch failed with status={final.status}. Errors: {errs[:3]}")
        raise RuntimeError(f"Batch finished with status={final.status}")
    
    out_id = getattr(final, "output_file_id", None)
    if not out_id:
        raise RuntimeError("Completed batch missing output_file_id")
    
    out_records = _download_output_file(out_id)
    
    # Extract text responses from records
    results: Dict[str, str] = {}
    for r in out_records:
        cid = r.get("custom_id")
        resp = (r.get("response") or {})
        status = resp.get("status_code")
        body = resp.get("body") or {}
        if status != 200:
            if verbose:
                print(f"[warning] Non-200 status for {cid}: {status}")
            results[cid] = f"ERROR: HTTP {status}"
        else:
            results[cid] = _extract_output_text(body)
    
    return results


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
    num_outputs: Union[int, List[int]] = 1,
    seeds: Optional[Union[int, List[Optional[int]], List[List[Optional[int]]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    include: Optional[List[str]] = None,
) -> Union[List[str], List[List[str]]]:
    """
    Submit one or many prompts as a single batch job.

    Parameters
    ----------
    tools : list[dict], optional
        Tools to enable (e.g., file_search). Example:
        [{"type": "file_search", "vector_store_ids": ["vs_123"], "max_num_results": 5}]
    include : list[str], optional
        Additional data to include in response (e.g., ["file_search_call.results"])

    Returns
    -------
    list[str] when every prompt requests a single completion, otherwise a
    list[list[str]] where each sub-list contains the completions for that input
    prompt in order.
    """
    items = prompts if isinstance(prompts, list) else [prompts]
    n = len(items)
    systems = _as_list(system_prompt, n, "system_prompt")
    counts = _broadcast_counts(num_outputs, n, "num_outputs")
    seeds_grid = _normalize_seed_grid(seeds, counts, "seeds")

    messages_per_item: List[List[Dict[str, str]]] = []
    for i in range(n):
        msgs: List[Dict[str, str]] = []
        if systems[i]:
            msgs.append({"role": "system", "content": systems[i]})  # type: ignore
        msgs.append({"role": "user", "content": items[i]})
        messages_per_item.append(msgs)

    lines, custom_ids, slot_map = _assemble_jsonl_lines(
        messages_per_item=messages_per_item,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        run_tag=run_tag,
        namespace="single",
        outputs_per_item=counts,
        seeds_per_item=seeds_grid,
        tools=tools,
        include=include,
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

    collected: List[List[str]] = [[""] * count for count in counts]
    for cid, (item_idx, variant_idx) in zip(custom_ids, slot_map):
        if cid not in outputs:
            raise KeyError(f"Missing output for custom_id {cid}")
        collected[item_idx][variant_idx] = outputs[cid]

    all_single = all(count == 1 for count in counts)
    if all_single:
        return [row[0] for row in collected]
    return [list(row) for row in collected]

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
    
    File search support: Pass tools parameter to enable file_search with vector stores.
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
        tools: Optional[List[Dict[str, Any]]] = None,
        include: Optional[List[str]] = None,
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
        self.tools = tools
        self.include = include

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
            "tools": self.tools,
            "include": self.include,
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
            tools=data.get("tools"),
            include=data.get("include"),
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
    def ask(
        self,
        user_prompts: Union[str, List[Optional[str]]],
        *,
        verbose: bool = True,
        num_outputs: Union[int, List[int]] = 1,
        seeds: Optional[Union[int, List[Optional[int]], List[List[Optional[int]]]]] = None,
        use_batch: bool = True,
        max_workers: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        include: Optional[List[str]] = None,
    ) -> Union[List[Optional[str]], List[Optional[List[str]]]]:
        """
        Add a user turn across threads, submit a batch or make synchronous calls,
        append assistant replies, and return the replies.

        Parameters
        ----------
        tools : list[dict], optional
            Tools to enable for this request. Overrides instance default if provided.
            Example: [{"type": "file_search", "vector_store_ids": ["vs_123"]}]
        include : list[str], optional
            Additional data to include. Overrides instance default if provided.
            Example: ["file_search_call.results"]
        
        Returns
        -------
        list[str | None] or list[list[str] | None]
            Replies aligned to thread indices (None where skipped).
        """
        # Convert to list format
        if isinstance(user_prompts, str):
            prompts: List[Optional[str]] = [user_prompts] * self.count
        else:
            prompts = user_prompts
            
        if len(prompts) != self.count:
            raise ValueError(f"user_prompts length {len(prompts)} must match count={self.count}")

        # Determine active threads
        active_indices = [i for i, p in enumerate(prompts) if p is not None]
        if not active_indices:
            return [None] * self.count  # no-op step, no increment

        counts = _broadcast_counts(num_outputs, self.count, "num_outputs")
        seeds_grid = _normalize_seed_grid(seeds, counts, "seeds")

        now = time.time()
        for i in active_indices:
            self._threads[i].append(_Turn("user", prompts[i] or "", now))

        effective_tools = tools if tools is not None else self.tools
        effective_include = include if include is not None else self.include

        if not use_batch:
            # Synchronous mode with parallel execution
            client = _get_client()
            collected: List[List[str]] = [[] for _ in active_indices]
            
            def make_completion(active_idx: int, variant_idx: int) -> Tuple[int, int, str]:
                """Helper to make a single completion call."""
                thread_idx = active_indices[active_idx]
                messages = [t.to_msg() for t in self._threads[thread_idx]]
                thread_seeds = seeds_grid[thread_idx]
                
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": self.max_output_tokens,
                }
                
                kwargs["reasoning_effort"] = self.reasoning_effort
                
                seed_value = thread_seeds[variant_idx]
                if seed_value is not None:
                    kwargs["seed"] = int(seed_value)
                
                if effective_tools is not None:
                    kwargs["tools"] = effective_tools
                if effective_include is not None:
                    kwargs["include"] = effective_include
                
                if verbose:
                    print(f"[sync] thread {thread_idx}, variant {variant_idx}")
                
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                return active_idx, variant_idx, text.strip()
            
            # Build list of all completion tasks
            tasks = []
            for active_idx in range(len(active_indices)):
                thread_idx = active_indices[active_idx]
                for variant_idx in range(counts[thread_idx]):
                    tasks.append((active_idx, variant_idx))
            
            # Execute all tasks in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(make_completion, active_idx, variant_idx): (active_idx, variant_idx)
                    for active_idx, variant_idx in tasks
                }
                
                for future in as_completed(futures):
                    active_idx, variant_idx, text = future.result()
                    collected[active_idx].append(text)
            
            # Ensure results are in correct order
            for active_idx in range(len(active_indices)):
                thread_idx = active_indices[active_idx]
                if len(collected[active_idx]) != counts[thread_idx]:
                    raise RuntimeError(f"Thread {thread_idx}: expected {counts[thread_idx]} variants, got {len(collected[active_idx])}")
            
            # Append assistant turns
            now2 = time.time()
            for active_idx, thread_idx in enumerate(active_indices):
                primary_reply = collected[active_idx][0]
                self._threads[thread_idx].append(_Turn("assistant", primary_reply, now2))
            
            # Build aligned result
            out: List[Optional[Union[str, List[str]]]] = [None] * self.count
            for active_idx, thread_idx in enumerate(active_indices):
                variants = collected[active_idx]
                if counts[thread_idx] == 1:
                    out[thread_idx] = variants[0]
                else:
                    out[thread_idx] = list(variants)
            
            self.step += 1
            return out

        # Batch mode - build payload only for active threads
        active_counts = [counts[i] for i in active_indices]
        active_seeds = [seeds_grid[i] for i in active_indices]

        messages_per_item: List[List[Dict[str, str]]] = []
        for i in active_indices:
            messages_per_item.append([t.to_msg() for t in self._threads[i]])

        ns = f"conv::step{self.step}"
        lines, custom_ids, slot_map = _assemble_jsonl_lines(
            messages_per_item=messages_per_item,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            run_tag=self.run_tag,
            namespace=ns,
            outputs_per_item=active_counts,
            seeds_per_item=active_seeds,
            tools=effective_tools,
            include=effective_include,
        )
        out_records = _run_batch_and_collect(lines, self.completion_window, verbose=verbose)

        # Map outputs back to thread indices
        slot_to_reply: Dict[int, List[str]] = {
            idx: [""] * counts[idx] for idx in active_indices
        }
        slot_lookup = {cid: slot for cid, slot in zip(custom_ids, slot_map)}
        
        for r in out_records:
            cid = r.get("custom_id")
            resp = (r.get("response") or {})
            status = resp.get("status_code")
            body = resp.get("body") or {}
            if status != 200:
                raise RuntimeError(f"Non-200 for {cid}: {status}")
            if cid not in slot_lookup:
                raise KeyError(f"Unexpected custom_id {cid} in batch output")
            slot_idx, variant_idx = slot_lookup[cid]
            thread_idx = active_indices[slot_idx]
            slot_to_reply[thread_idx][variant_idx] = _extract_output_text(body)

        # Append assistant turns
        now2 = time.time()
        for i in active_indices:
            reply = slot_to_reply[i][0]
            self._threads[i].append(_Turn("assistant", reply, now2))

        # Build aligned result (None where skipped)
        out: List[Optional[Union[str, List[str]]]] = [None] * self.count
        for i in active_indices:
            variants = slot_to_reply[i]
            if counts[i] == 1:
                out[i] = variants[0]
            else:
                out[i] = list(variants)

        self.step += 1
        return out

    def ask_map(
        self,
        prompt_map: Dict[int, str],
        *,
        verbose: bool = True,
        num_outputs: Union[int, List[int]] = 1,
        seeds: Optional[Union[int, List[Optional[int]], List[List[Optional[int]]]]] = None,
        use_batch: bool = True,
        max_workers: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[int, Union[str, List[str]]]:
        """
        Convenience wrapper: provide {thread_index: prompt}. Returns {thread_index: reply}.
        """
        masked: List[Optional[str]] = [None] * self.count
        for i, p in prompt_map.items():
            if i < 0 or i >= self.count:
                raise IndexError(f"thread index out of range: {i}")
            masked[i] = p
        replies = self.ask(
            masked,
            verbose=verbose,
            num_outputs=num_outputs,
            seeds=seeds,
            use_batch=use_batch,
            max_workers=max_workers,
            tools=tools,
            include=include,
        )
        return {i: (replies[i] if replies[i] is not None else "") for i in prompt_map.keys()}

    def recover_and_apply(
        self,
        batch_id: str,
        user_prompts: Union[str, List[str]],
        *,
        poll_interval: float = 5.0,
        verbose: bool = True,
    ) -> List[str]:
        """
        Recover a batch result and apply it to conversation history.
        
        Use this if you submitted a batch but lost the results. You must provide
        the same user_prompts that were originally sent so they can be added to
        the conversation history before the assistant responses.
        
        Parameters
        ----------
        batch_id : str
            The batch job ID to recover
        user_prompts : str | list[str]
            The user prompts that were sent (must match count)
        poll_interval : float
            Seconds between status checks
        verbose : bool
            Print status updates
            
        Returns
        -------
        list[str]
            The assistant replies (one per thread)
            
        Notes
        -----
        This assumes the batch was created with a single output per thread.
        The step counter is incremented.
        """
        prompts = user_prompts if isinstance(user_prompts, list) else [user_prompts] * self.count
        if len(prompts) != self.count:
            raise ValueError(f"user_prompts length {len(prompts)} must match count={self.count}")
        
        # Recover batch output - now returns Dict[str, str] mapping custom_id to text
        results_by_cid = recover_batch_result(batch_id, poll_interval=poll_interval, verbose=verbose)
        
        # Parse custom_ids to extract thread indices and build reply mapping
        by_thread: Dict[int, str] = {}
        for cid, text in results_by_cid.items():
            # Parse custom_id format: "conv::stepX::run_tag::thread_idx" or similar
            parts = cid.split("::")
            if len(parts) >= 4:
                try:
                    # Format is namespace::stepX::run_tag::thread_idx
                    thread_idx_str = parts[3].replace("v", "").split("v")[0]
                    thread_idx = int(thread_idx_str)
                    by_thread[thread_idx] = text
                except (ValueError, IndexError):
                    raise ValueError(f"Cannot parse thread index from custom_id: {cid}")
            else:
                raise ValueError(f"Unexpected custom_id format: {cid}")
        
        if len(by_thread) != self.count:
            raise RuntimeError(f"Expected {self.count} responses, got {len(by_thread)}")
        
        # Add user and assistant turns to history
        now = time.time()
        for i in range(self.count):
            if i not in by_thread:
                raise KeyError(f"Missing response for thread {i}")
            self._threads[i].append(_Turn("user", prompts[i], now))
            self._threads[i].append(_Turn("assistant", by_thread[i], now))
        
        self.step += 1
        return [by_thread[i] for i in range(self.count)]
