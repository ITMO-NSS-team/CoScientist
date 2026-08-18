"""Start a demo replay in the running UI, and print what to open.

Usage:
    python scripts/demo_replay.py <recording> [--speed 120] [--gap 2.5]

`<recording>` is a directory holding the session's `research_active.json` — a
bundle from `collect_session.py`, or a session directory under
`graph_runs/sessions/<user>/<session>`.

Speed compresses the recorded intervals; `--gap` caps any single pause, which
matters more than the ratio: a real run contains forty-minute waits for a
sandbox, and at any honest speed those are still dead air in front of an
audience.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--speed", type=float, default=120)
    ap.add_argument("--gap", type=float, default=2.5,
                    help="longest pause between two steps, seconds")
    ap.add_argument("--title", default="Recorded study (replay)")
    ap.add_argument("--user", default="", help="user id; the first one by default")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    payload = json.dumps({
        "bundle": args.recording, "speed": args.speed, "max_gap": args.gap,
        "title": args.title, "user_id": args.user,
    }).encode()
    request = urllib.request.Request(
        f"{args.base.rstrip('/')}/api/demo/replay", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"refused ({exc.code}): {exc.read().decode()[:300]}")
        return 1
    except urllib.error.URLError as exc:
        print(f"cannot reach {args.base}: {exc.reason}\n"
              f"start the server first: python -m CoScientist web")
        return 1

    print(f"replaying {args.recording}")
    print(f"  {data['events']} events, {data['nodes']} graph nodes, {data['speed']:g}x")
    print(f"\nopen and record this:\n  {args.base.rstrip('/')}{data['open']}")
    print("\nthe graph view is under the 'research · slide' selector; "
          "set SLIDE_LANG=en before starting the server for English labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
