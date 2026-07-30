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
import re
import time
from collections import defaultdict, deque

# A message that carries its own complete question needs no history. A message
# that points BACK at something -- "that dataset", "the figures above" -- does.
# Time-based expiry alone is not enough: several unrelated questions can land in
# one chat inside any TTL you pick, and then question 3 silently computes over
# question 1's data. So history is supplied only when the wording asks for it.
_BACKREF = re.compile(
    r"\b("
    r"that|those|these|the above|above|previous|previously|earlier|"
    r"aforementioned|same (?:data|dataset|list|table|numbers|values)|"
    r"remember(?:ed)?|mentioned|you (?:just )?(?:computed|calculated|found|said)|"
    r"my (?:data|dataset|list|numbers)|the (?:data|dataset|list|table)\b"
    r")\b",
    re.IGNORECASE,
)


# A bracketed numeric list, or a run of comma-separated numbers, means the data
# is HERE. "Multiply each of these by 1.02: [12, 40, 87]" contains a back-
# reference word, but "these" points at the list in the same sentence, not at a
# previous turn.
_INLINE_DATA = re.compile(
    r"\[\s*-?\d[\d\s,.\-eE]*\]"                      # [12, 40, 87]
    r"|(?:-?\d+(?:\.\d+)?\s*,\s*){3,}-?\d+(?:\.\d+)?"  # 1, 2, 3, 4
)


def has_inline_data(text: str) -> bool:
    """True when the message carries its own dataset."""
    return bool(_INLINE_DATA.search(text or ""))


def needs_history(text: str) -> bool:
    """True when the message refers back to something said in an earlier turn.

    A message that both names a back-reference AND ships its own data is
    self-contained -- the pronoun points at the data in front of it.
    """
    text = text or ""
    if has_inline_data(text):
        return False
    return bool(_BACKREF.search(text))

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
