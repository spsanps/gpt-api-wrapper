"""OpenAI provider implementation."""
from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from typing import Dict, Iterable, Iterator, List, Sequence

from .base import BaseProvider, Message, register_provider

logger = logging.getLogger(__name__)


def _extract_output_text_from_resp(resp) -> str:
    if resp is None:
        return ""

    # SDK object path
    if hasattr(resp, "output_text") and getattr(resp, "output_text"):
        return str(getattr(resp, "output_text")).strip()

    # dict shape (batch body)
    if isinstance(resp, dict) and resp.get("output_text"):
        return str(resp.get("output_text")).strip()

    pieces: List[str] = []
    output = None
    if isinstance(resp, dict):
        output = resp.get("output")
    else:
        output = getattr(resp, "output", None)

    for item in output or []:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "message":
            contents = item.get("content") if isinstance(item, dict) else getattr(item, "content", [])
            for c in contents or []:
                text_val = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
                if text_val:
                    pieces.append(text_val)
    return "".join(pieces).strip()


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_streaming = True

    def _create_client(self):  # pragma: no cover - thin wrapper
        from openai import OpenAI

        return OpenAI()

    # -- single / conversation -------------------------------------------------
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

        if stream:
            from contextlib import contextmanager

            @contextmanager
            def _streaming():
                s = client.responses.stream(
                    model=model,
                    input=list(messages),
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=max_output_tokens,
                )
                try:
                    yield s
                finally:
                    s.close()

            def _generator() -> Iterator[str]:
                pieces: List[str] = []
                with _streaming() as stream_obj:
                    for event in stream_obj:
                        if event.type == "response.output_text.delta":
                            chunk = event.delta
                            if chunk:
                                pieces.append(chunk)
                                yield chunk
                    stream_obj.finalize()
                    full = "".join(pieces).strip()
                    logger.debug("OpenAI stream finalised with %d chars", len(full))
            return _generator()

        resp = client.responses.create(
            model=model,
            input=list(messages),
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
        )
        text = _extract_output_text_from_resp(resp)
        logger.debug("OpenAI response length=%d", len(text))
        return text

    # -- batch -----------------------------------------------------------------
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
        if not batched_messages:
            return []

        client = self.ensure_client()

        lines: List[Dict[str, object]] = []
        custom_ids: List[str] = []
        run_tag = uuid.uuid4().hex[:10]

        for idx, msgs in enumerate(batched_messages):
            custom_id = f"job::{run_tag}::{idx}"
            body = {
                "model": model,
                "input": list(msgs),
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": int(max_output_tokens),
            }
            lines.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
            )
            custom_ids.append(custom_id)

        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        with tmp_file as fh:
            for line in lines:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp_path = tmp_file.name

        try:
            in_file = client.files.create(file=open(tmp_path, "rb"), purpose="batch")
            batch = client.batches.create(
                input_file_id=in_file.id,
                endpoint="/v1/responses",
                completion_window=completion_window,
            )
            batch_id = batch.id
            logger.info("Submitted OpenAI batch %s with %d items", batch_id, len(lines))

            try:
                from tqdm.auto import tqdm  # type: ignore
            except Exception:  # pragma: no cover - UI component
                def _poll_iterator():
                    while True:
                        yield None
                iterator = _poll_iterator()
            else:  # pragma: no cover - UI component
                iterator = tqdm(desc=f"openai batch {batch_id}", disable=not verbose)

            status = None
            try:
                while True:
                    batch_obj = client.batches.retrieve(batch_id)
                    new_status = getattr(batch_obj, "status", None)
                    if new_status != status:
                        logger.info("Batch %s status -> %s", batch_id, new_status)
                        status = new_status
                    if new_status not in {"validating", "in_progress", "finalizing"}:
                        break
                    time.sleep(5)
                    if hasattr(iterator, "update"):
                        iterator.update(1)
                if hasattr(iterator, "close"):
                    iterator.close()
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()

            if getattr(batch_obj, "status", None) != "completed":
                err_id = getattr(batch_obj, "error_file_id", None) or getattr(batch_obj, "errors_file_id", None)
                if err_id:
                    err_stream = client.files.content(err_id)
                    raw = err_stream.read().decode("utf-8")
                    raise RuntimeError(f"Batch {batch_id} failed: {raw}")
                raise RuntimeError(f"Batch {batch_id} finished with status={batch_obj.status}")

            out_file_id = getattr(batch_obj, "output_file_id", None)
            if not out_file_id:
                raise RuntimeError(f"Batch {batch_id} missing output_file_id")

            content_stream = client.files.content(out_file_id)
            decoded = content_stream.read().decode("utf-8")
            outputs: Dict[str, str] = {}
            for line in decoded.splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                cid = data.get("custom_id")
                resp = (data.get("response") or {})
                status_code = resp.get("status_code")
                if status_code != 200:
                    raise RuntimeError(f"Non-200 response in batch: {cid}: {status_code}")
                body = resp.get("body") or {}
                outputs[cid] = _extract_output_text_from_resp(body)

            return [outputs[cid] for cid in custom_ids]
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


register_provider(OpenAIProvider())
