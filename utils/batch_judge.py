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

    def _run_sync(self) -> dict[str, Optional[dict]]:
        """Blocking: upload, create batch, poll, download, parse. Runs in a thread."""
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

        # No client-side timeout: the OpenAI batch SLA is up to 24h and the user
        # accepts a long wait. Poll until the batch reaches a terminal state.
        waited = 0
        while batch.status not in _TERMINAL:
            time.sleep(self.poll_seconds)
            waited += self.poll_seconds
            batch = client.batches.retrieve(batch.id)
            counts = getattr(batch, "request_counts", None)
            logger.info(
                "Judge batch %s: status=%s, %ss elapsed%s",
                batch.id, batch.status, waited,
                f", {counts.completed}/{counts.total} done" if counts else "",
            )

        if batch.status != "completed":
            raise RuntimeError(f"Judge batch {batch.id} ended in state '{batch.status}'")

        out_text = client.files.content(batch.output_file_id).text
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
        logger.info("Judge batch %s complete: %d/%d parsed", batch.id, len(results), self.count)
        return results

    async def run(self) -> dict[str, Optional[dict]]:
        """Submit + await the batch. Returns {custom_id: payload or None}."""
        if not self._requests:
            return {}
        return await asyncio.to_thread(self._run_sync)
