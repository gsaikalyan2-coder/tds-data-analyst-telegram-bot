#!/usr/bin/env python3
"""Offline smoke test: replays the worked example from the assignment through
the real formatting path, with a stubbed agent. Proves the bytes that would go
to Telegram are exactly one JSON object.

    python3 scripts/selftest.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer_format import (build_reply, coerce_answer,
                               enforce_single_json_object,
                               extract_response_template)

CASES = [
    (
        'Which state has the highest maternal mortality rate based on MOSPI data? '
        'Reply with ONLY this JSON object and nothing else: '
        '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL>"}',
        {"state": "Assam"},
    ),
    (
        'Forecast for these inputs: [10, 20]. Reply with ONLY {"values": [<numbers>]}.',
        {"values": [10.2, 20.4]},
    ),
    (
        'Reply with ONLY a JSON object like {"state": "<state name>"}',
        {"state": "Kerala"},
    ),
]

LOG_URL = "https://your-service.onrender.com/run.jsonl"
failures = 0
for message, model_answer in CASES:
    template = extract_response_template(message)
    answer = coerce_answer(model_answer, template)
    reply = enforce_single_json_object(build_reply(answer, LOG_URL, template))
    try:
        parsed = json.loads(reply)
        assert isinstance(parsed, dict)
        assert reply.strip() == reply and "\n" not in reply
        print(f"PASS  {reply}")
    except Exception as exc:
        failures += 1
        print(f"FAIL  {exc}  ->  {reply!r}")

print("\nall good" if not failures else f"\n{failures} FAILURES")
sys.exit(1 if failures else 0)
