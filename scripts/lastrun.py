#!/usr/bin/env python3
"""Pretty-print the most recent agent run from logs/run.jsonl.

The rolling log is append-only across every question the bot has ever
answered, so `type logs\\run.jsonl` quickly becomes unreadable. This shows just
the last run (or the last N), collapsing each event to the one line that
matters when you are debugging an answer.

    python scripts/lastrun.py           # last run
    python scripts/lastrun.py -n 3      # last 3 runs
    python scripts/lastrun.py --full    # don't truncate tool arguments
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "run.jsonl"


def clip(text: object, n: int, full: bool) -> str:
    s = str(text or "").replace("\n", " | ")
    return s if full or len(s) <= n else s[:n] + "..."


def render(run: list[dict], full: bool) -> None:
    head = run[0]
    print(f"\n=== run {head['run_id']}  ({len(run)} events) ===")
    for r in run:
        e = r.get("event")
        t = f"[{r.get('elapsed_s', 0):>7}s]"

        if e == "message_received":
            print(f"{t} QUESTION  {clip(r.get('text'), 160, full)}")
        elif e == "template_extracted":
            print(f"{t} TEMPLATE  {json.dumps(r.get('template'))}")
        elif e == "tool_call":
            a = r.get("arguments", {}) or {}
            detail = (
                a.get("query")
                or (a.get("url", "") + (f"   find={a['find']!r}" if a.get("find") else ""))
                or a.get("code", "")
            )
            print(f"{t} CALL      {r.get('tool')}: {clip(detail, 220, full)}")
        elif e == "tool_result":
            o = r.get("output", {}) or {}
            if isinstance(o, dict) and o.get("provider"):
                print(f"{t} RESULT    search via {o.get('provider')}"
                      f"{' (cached)' if o.get('cached') else ''}")
                for x in (o.get("results") or [])[:4]:
                    print(f"             - {x.get('url')}")
            elif isinstance(o, dict) and o.get("matched_pages") is not None:
                print(f"{t} RESULT    pdf {o.get('pages')} pages, "
                      f"{o.get('matched_pages')} matched")
                for m in (o.get("results") or [])[:3]:
                    rows = m.get("first_table_rows") or []
                    print(f"             p{m.get('page')} header={rows[0] if rows else '(no table)'}")
            else:
                body = o.get("stdout") or o.get("error") or o.get("stderr") or ""
                print(f"{t} RESULT    ok={r.get('ok')}  {clip(body, 240, full)}")
        elif e == "model_plan":
            print(f"{t} PLAN      {clip(r.get('content'), 200, full)}")
        elif e == "final_answer_tool":
            args = r.get("arguments", {}) or {}
            print(f"{t} FINAL")
            print(f"             answer    = {args.get('answer_json')}")
            print(f"             source    = {r.get('source')}")
            print(f"             indicator = {r.get('indicator')}")
            print(f"             period    = {r.get('reference_period')}")
            print(f"             reasoning = {clip(args.get('reasoning'), 240, full)}")
        elif e == "warning_unsourced_answer":
            print(f"{t} *** UNSOURCED ANSWER -- the agent could not retrieve data ***")
        elif e in {"no_answer_after_loop", "deadline_pressure",
                   "answer_skeleton_fallback", "model_fallback"}:
            print(f"{t} !! {e}  {clip(r.get('detail') or r.get('remaining_s') or '', 120, full)}")
        elif e == "error":
            print(f"{t} ERROR     {r.get('where')}: {clip(r.get('detail'), 200, full)}")
        elif e == "final_answer":
            print(f"{t} SENT      {clip(r.get('reply_text'), 200, full)}")

    # ---- quick verdict --------------------------------------------------
    final = next((r for r in reversed(run) if r.get("event") == "final_answer_tool"), None)
    warnings = [r for r in run if r.get("event") in {
        "warning_unsourced_answer", "no_answer_after_loop",
        "deadline_pressure", "answer_skeleton_fallback"}]
    print("\n  VERDICT:", end=" ")
    if not final:
        print("no final_answer tool call -- answer came from a fallback path")
    elif warnings:
        print(f"answered, but with {len(warnings)} warning(s) above -- treat as unverified")
    elif str(final.get("source", "")).startswith("http"):
        print(f"verified against {final.get('source')}")
        print(f"           indicator '{final.get('indicator')}', period '{final.get('reference_period')}'")
        print("           -> check that indicator really names the quantity asked for")
    else:
        print(f"source = {final.get('source')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--runs", type=int, default=1, help="how many recent runs to show")
    ap.add_argument("--full", action="store_true", help="do not truncate long fields")
    args = ap.parse_args()

    if not LOG.exists():
        print(f"no log yet at {LOG}")
        return 1

    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a torn last line during a live write
    if not rows:
        print("log is empty")
        return 1

    order: list[str] = []
    for r in rows:
        rid = r.get("run_id")
        if rid and rid not in order:
            order.append(rid)
    for rid in order[-args.runs:]:
        render([r for r in rows if r.get("run_id") == rid], args.full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
