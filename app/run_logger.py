"""Append-only JSONL run logger.

Two files are maintained for every run:

  logs/run.jsonl            -- rolling cumulative log; this is what `log_url`
                               points at by default (graders wget it)
  logs/runs/<run_id>.jsonl  -- just this one run, for per-question review

Every line is exactly one JSON object.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_MAX_FIELD_CHARS = 8000


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + f"...[truncated {len(value) - _MAX_FIELD_CHARS} chars]"
    if isinstance(value, list):
        return [_truncate(v) for v in value[:50]]
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


class RunLogger:
    """One instance per incoming Telegram message (i.e. per agent run)."""

    def __init__(self, log_dir: str, public_base_url: str, chat_id: int | str = "-"):
        self.run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.chat_id = str(chat_id)
        self.started = time.time()
        self.log_dir = Path(log_dir)
        self.runs_dir = self.log_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_path = self.log_dir / "run.jsonl"
        self.run_path = self.runs_dir / f"{self.run_id}.jsonl"
        self.public_base_url = public_base_url.rstrip("/")
        self._step = 0

    # ---------------------------------------------------------------- urls
    @property
    def log_url(self) -> str:
        """Public, wget-able URL to the cumulative JSONL log."""
        return f"{self.public_base_url}/run.jsonl"

    @property
    def run_log_url(self) -> str:
        """Public, wget-able URL to just this run."""
        return f"{self.public_base_url}/logs/{self.run_id}.jsonl"

    # --------------------------------------------------------------- write
    def event(self, kind: str, **fields: Any) -> None:
        self._step += 1
        record = {
            "run_id": self.run_id,
            "step": self._step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "elapsed_s": round(time.time() - self.started, 3),
            "chat_id": self.chat_id,
            "event": kind,
            **{k: _truncate(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _LOCK:
            for path in (self.run_path, self.rolling_path):
                try:
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(line)
                except OSError:
                    pass  # logging must never break the answer path

    # Convenience wrappers -------------------------------------------------
    def message_received(self, text: str, turn_index: int) -> None:
        self.event("message_received", turn_index=turn_index, text=text)

    def plan(self, content: str) -> None:
        self.event("model_plan", content=content)

    def tool_call(self, name: str, args: dict) -> None:
        self.event("tool_call", tool=name, arguments=args)

    def tool_result(self, name: str, ok: bool, output: Any) -> None:
        self.event("tool_result", tool=name, ok=ok, output=output)

    def error(self, where: str, detail: str) -> None:
        self.event("error", where=where, detail=detail)

    def final(self, answer: Any, reply_text: str) -> None:
        self.event("final_answer", answer=answer, reply_text=reply_text,
                   total_seconds=round(time.time() - self.started, 3))


def ensure_log_files(log_dir: str) -> None:
    """Create logs/run.jsonl up front so log_url is never a 404, even before
    the first question arrives (graders may probe it)."""
    d = Path(log_dir)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    rolling = d / "run.jsonl"
    if not rolling.exists():
        boot = {
            "run_id": "boot",
            "step": 0,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "event": "service_started",
            "note": "TDS P1 data-analyst Telegram bot log. One JSON object per line.",
        }
        rolling.write_text(json.dumps(boot) + "\n", encoding="utf-8")
