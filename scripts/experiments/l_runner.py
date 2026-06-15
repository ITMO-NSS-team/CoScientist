#!/usr/bin/env python
"""dataset_L runner: complex multi-task queries through the LIVE pipeline.

Like ab_runner.py but reads CoScientist/dataset_L.xlsx ('content' = multi-task query,
'task 1..5' = ground-truth sub-tasks kept for per-task evaluation). Forces qwen and
records session_id -> Opik trace_id.

Flags:
  --condition L    session/label prefix
  --no-action-critic   disable the per-action pre_action_critique (test "plan-critic only")
  --limit N --cap S

Usage: python scripts/experiments/l_runner.py --condition L1 [--no-action-critic] [--limit 5] [--cap 600]
"""
import argparse
import asyncio
import datetime
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("LLM__MAIN_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
os.environ.setdefault("HITL__HEADLESS_AUTO_APPROVE", "true")  # test the chosen HITL fix
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/opik_eval"))
from trace_locator import record_run  # noqa: E402
import openpyxl  # noqa: E402


def load_queries(n: int) -> list[dict]:
    wb = openpyxl.load_workbook(REPO / "CoScientist/dataset_L.xlsx", read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    seen, out = set(), []
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        q = str(d.get("content") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        tasks = [str(d.get(f"task {i}")).strip() for i in range(1, 6) if d.get(f"task {i}")]
        out.append({"query": q, "tasks": tasks, "case": d.get("case")})
        if len(out) >= n:
            break
    return out


async def run_one(item: dict, session_id: str, cap: int) -> dict:
    from CoScientist.main import CoScientistManager

    mgr = CoScientistManager(session_id=session_id)
    t0 = time.time()
    err, resp = None, ""
    try:
        resp = await asyncio.wait_for(mgr.run(item["query"], verbose=False), timeout=cap)
    except asyncio.TimeoutError:
        err = f"timeout>{cap}s"
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        try:
            await mgr.close()
        except Exception:
            pass
    return {
        "query": item["query"][:90],
        "tasks": item["tasks"],
        "session_id": session_id,
        "duration_s": round(time.time() - t0, 1),
        "resp_len": len(resp or ""),
        "resp_head": (resp or "")[:300],
        "has_s3": ("X-Amz-Signature" in (resp or "")) or ("10.32.1.114:9000" in (resp or "")),
        "error": err,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--no-action-critic", action="store_true", help="disable the per-action critic entirely")
    ap.add_argument("--plan-critic", action="store_true", help="critic reviews the PLAN once (first delegation), not every action")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--cap", type=int, default=600)
    args = ap.parse_args()

    import CoScientist.agents.agents as A
    from CoScientist.config.settings import get_settings
    if args.no_action_critic:
        A.orchestrator_agent.after_model_callback = None
    if args.plan_critic:
        get_settings().orchestrator.plan_critic_only = True

    stamp = datetime.datetime.now().strftime("%H%M%S")
    crit = ("NO-action-critic" if args.no_action_critic
            else "plan-critic-only" if args.plan_critic else "with-action-critic")
    print(f"=== L run · condition={args.condition} · {crit} · model={os.environ['LLM__MAIN_MODEL']} "
          f"· auto_approve={os.environ.get('HITL__HEADLESS_AUTO_APPROVE')} · {args.limit} queries · cap={args.cap}s ===",
          flush=True)
    items = load_queries(args.limit)
    runs = []
    for i, it in enumerate(items):
        sid = f"l_{args.condition}_{i:02d}_{stamp}"
        print(f"[{i + 1}/{len(items)}] sid={sid} :: {it['query'][:60]}", flush=True)
        r = await run_one(it, sid, args.cap)
        print(f"    -> {r['duration_s']}s len={r['resp_len']} s3={r['has_s3']} err={r['error']}", flush=True)
        entry = await asyncio.to_thread(record_run, sid, query=it["query"], condition=args.condition,
                                        model=os.environ["LLM__MAIN_MODEL"])
        r["trace_id"] = entry["trace_id"] if entry else None
        print(f"    trace={r['trace_id']}", flush=True)
        runs.append(r)

    outdir = REPO / "scripts/experiments/results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"l_{args.condition}_{datetime.date.today().isoformat()}_{stamp}.json"
    out.write_text(json.dumps(
        {"condition": args.condition, "no_action_critic": args.no_action_critic,
         "model": os.environ["LLM__MAIN_MODEL"], "runs": runs}, ensure_ascii=False, indent=1))
    ok = sum(1 for r in runs if not r["error"] and r["resp_len"] > 0)
    s3 = sum(1 for r in runs if r["has_s3"])
    print(f"\n=== done: {ok}/{len(runs)} responded · {s3}/{len(runs)} with S3 link · saved {out} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
