"""OpenAI Batch API collector for the benchmark LLM-as-a-judge call.

Opt-in via ``RAG_JUDGE_BATCH=true`` (and an OpenAI ``EVAL_MODEL``). The
benchmark registers every judge prompt during its first pass, then submits a
single batch to the ``/v1/chat/completions`` endpoint (50% cheaper than
synchronous calls), polls to completion, and returns
``{custom_id: parsed_payload}``. The caller falls back to the synchronous
per-query judge if anything here raises.

The OpenAI SDK calls are synchronous, so they run in worker threads via
``asyncio.to_thread`` to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Optional

from utils.parsers import clean_and_unwrap_json

logger = logging.getLogger("PreHypo")

# Terminal batch states per the OpenAI Batch API.
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchJudge:
    """Collect judge prompts, run them as one OpenAI batch, map results back."""

    def __init__(
        self,
        model: str,
        api_key: str,
        poll_seconds: int = 15,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.poll_seconds = max(2, int(poll_seconds))
        self._requests: list[tuple[str, str]] = []

    def register(self, custom_id: str, prompt: str) -> None:
        self._requests.append((str(custom_id), prompt))

    @property
    def count(self) -> int:
        return len(self._requests)

    def _build_jsonl(self) -> bytes:
        buf = io.BytesIO()
        for custom_id, prompt in self._requests:
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            }
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        return buf.getvalue()

    def _submit_sync(self) -> Optional[str]:
        """Upload the JSONL and create the batch. Returns the batch id (no poll)."""
        from openai import OpenAI  # local import so the dep is only needed when used

        client = OpenAI(api_key=self.api_key)
        upload = client.files.create(
            file=("judge_batch.jsonl", self._build_jsonl()),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info("Judge batch submitted: id=%s, %d requests", batch.id, self.count)
        return batch.id

    async def submit(self) -> Optional[str]:
        """Submit the batch without waiting. Returns the batch id (or None when
        there is nothing to judge). Use `resolve_batches`/`poll_and_fetch` later
        to retrieve the results — the work runs asynchronously on OpenAI's side."""
        if not self._requests:
            return None
        return await asyncio.to_thread(self._submit_sync)

    def _run_sync(self) -> dict[str, Optional[dict]]:
        """Blocking: submit, poll, download, parse. Runs in a thread."""
        batch_id = self._submit_sync()
        return poll_and_fetch(self.api_key, batch_id, self.poll_seconds)

    async def run(self) -> dict[str, Optional[dict]]:
        """Submit + await the batch. Returns {custom_id: payload or None}."""
        if not self._requests:
            return {}
        return await asyncio.to_thread(self._run_sync)


def _parse_batch_output(out_text: str, total: Optional[int] = None) -> dict[str, Optional[dict]]:
    """Parse a batch output JSONL into {custom_id: parsed_payload}."""
    results: dict[str, Optional[dict]] = {}
    for raw in out_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            custom_id = row.get("custom_id")
            content = row["response"]["body"]["choices"][0]["message"]["content"]
            results[custom_id] = json.loads(clean_and_unwrap_json(content))
        except Exception as exc:  # one bad line shouldn't sink the batch
            logger.warning("Batch judge: could not parse output line: %s", exc)
    logger.info("Judge batch parsed: %d%s payloads", len(results), f"/{total}" if total else "")
    return results


def poll_and_fetch(
    api_key: str,
    batch_id: str,
    poll_seconds: int = 15,
    client=None,
) -> dict[str, Optional[dict]]:
    """Block until `batch_id` reaches a terminal state, then download + parse.

    No client-side timeout — the OpenAI batch SLA is up to 24h. Raises if the
    batch ends in any state other than ``completed``.
    """
    from openai import OpenAI

    client = client or OpenAI(api_key=api_key)
    poll_seconds = max(2, int(poll_seconds))
    batch = client.batches.retrieve(batch_id)
    waited = 0
    while batch.status not in _TERMINAL:
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        logger.info(
            "Judge batch %s: status=%s, %ss elapsed%s",
            batch.id, batch.status, waited,
            f", {counts.completed}/{counts.total} done" if counts else "",
        )
    if batch.status != "completed":
        raise RuntimeError(f"Judge batch {batch_id} ended in state '{batch.status}'")
    out_text = client.files.content(batch.output_file_id).text
    return _parse_batch_output(out_text)


def resolve_batches(
    api_key: str,
    batch_ids: list[str],
    poll_seconds: int = 15,
) -> dict[str, dict[str, Optional[dict]]]:
    """Poll several batches concurrently. Returns {batch_id: {custom_id: payload}}.

    A batch that fails/expires (or whose download errors) maps to an empty dict
    and is logged — the caller decides how to treat the unresolved rows.
    """
    from concurrent.futures import ThreadPoolExecutor

    ids = [b for b in dict.fromkeys(batch_ids) if b]
    out: dict[str, dict[str, Optional[dict]]] = {}
    if not ids:
        return out

    def _one(bid: str) -> tuple[str, dict[str, Optional[dict]]]:
        try:
            return bid, poll_and_fetch(api_key, bid, poll_seconds)
        except Exception as exc:
            logger.error("Judge batch %s did not resolve: %s", bid, exc)
            return bid, {}

    with ThreadPoolExecutor(max_workers=max(1, len(ids))) as pool:
        for bid, res in pool.map(_one, ids):
            out[bid] = res
    return out
