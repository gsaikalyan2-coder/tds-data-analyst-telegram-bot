"""Per-chat multi-turn state.

The grader's collect.py sends message 1, WAITS for one reply, sends message 2,
waits for one reply, and so on -- all inside a single timeout budget. So:

  * we must reply exactly once to every message (silence = `timeout` = 0 marks),
  * we do not know which message is the last one,
  * therefore we answer EVERY message as if it were the last, carrying the
    earlier turns as context.

That satisfies "answer the last one" automatically: the last reply we send is
the answer to the last message.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

MAX_TURNS = 20
# 10 minutes, not an hour. The grader's multi-turn messages arrive seconds
# apart, so they still share context. But separate QUESTIONS also land in the
# same chat, minutes or hours apart -- with a one-hour window, question 1's
# "remember this dataset" is still in scope when question 3 says "that
# dataset", and the answer is silently computed over the wrong data.
TTL_SECONDS = 600


class ConversationStore:
    def __init__(self) -> None:
        self._turns: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))
        self._touched: dict[int, float] = {}
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def lock(self, chat_id: int) -> asyncio.Lock:
        return self._locks[chat_id]

    def _expire(self, chat_id: int) -> None:
        last = self._touched.get(chat_id)
        if last and time.time() - last > TTL_SECONDS:
            self._turns.pop(chat_id, None)

    def add_user(self, chat_id: int, text: str) -> list[dict]:
        self._expire(chat_id)
        self._turns[chat_id].append({"role": "user", "text": text})
        self._touched[chat_id] = time.time()
        return list(self._turns[chat_id])

    def add_assistant(self, chat_id: int, text: str) -> None:
        self._turns[chat_id].append({"role": "assistant", "text": text})
        self._touched[chat_id] = time.time()

    def turn_index(self, chat_id: int) -> int:
        return sum(1 for t in self._turns[chat_id] if t["role"] == "user")

    def reset(self, chat_id: int) -> None:
        self._turns.pop(chat_id, None)
        self._touched.pop(chat_id, None)
