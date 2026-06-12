#!/usr/bin/env python
"""A/B(/C) runner: run dataset_S queries through the LIVE CoScientist pipeline.

Each run uses a unique session_id = ab_<condition>_<i>_<stamp>, which becomes the
Opik trace's thread_id — so traces are discriminable by condition afterwards.
Model is forced to qwen (set BEFORE importing CoScientist so the OpikTracer metadata
and all agents pick it up). Run each condition as a SEPARATE process so a prompt edit
between conditions takes effect (the orchestrator agent is built at import).

Usage: python scripts/experiments/ab_runner.py --condition A [--limit 10] [--cap 480]
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
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Verbatim dataset_S drug-design tasks — first 4 are ONE PER DOMAIN (used for --limit 4):
#   GSK-3β/Alzheimer, KRAS-G12C/cancer, BTK/sclerosis, STAT3/drug-resistance.
QUERIES = [
    "Generate GSK-3beta inhibitors with high activity",
    "Generate inhibitors of KRAS protein with G12C mutation. The inhibitors should be selective, meaning they should not bind with HRAS and NRAS proteins.",
    "Generate high activity tyrosine-protein kinase BTK inhibitors",
    "Can you suggest molecules that inhibit signal transducer and activator of transcription 3 (STAT3) with water solubility greater than 60 ug/mL and inhibitory ability to P450 CYP1A2?",
    "Can you suggest molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and the ability to cross the BBB?",
    "Generate GSK-3beta inhibitors with high docking score and low brain-blood barrier permeability",
    "Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer.",
    "Can you suggest molecules that inhibit ABL tyrosine-protein kinase with an LD50 toxicity of 501 mg/kg or more and a half-life of 4 hours?",
]


async def run_one(query: str, session_id: str, cap: int) -> dict:
    from CoScientist.main import CoScientistManager

    mgr = CoScientistManager(session_id=session_id)
    t0 = time.time()
    err = None
    resp = ""
    try:
        resp = await asyncio.wait_for(mgr.run(query, verbose=False), timeout=cap)
    except asyncio.TimeoutError:
        err = f"timeout>{cap}s"
    except Exception as exc:  # keep the batch going
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        try:
            await mgr.close()
        except Exception:
            pass
    return {
        "query": query[:90],
        "session_id": session_id,
        "duration_s": round(time.time() - t0, 1),
        "resp_len": len(resp or ""),
        "resp_head": (resp or "")[:200],
        "error": err,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--cap", type=int, default=480, help="per-run wall-clock cap (s)")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%H%M%S")
    print(f"=== A/B run · condition={args.condition} · model={os.environ['LLM__MAIN_MODEL']} "
          f"· {args.limit} queries · cap={args.cap}s ===", flush=True)
    runs = []
    for i, q in enumerate(QUERIES[: args.limit]):
        sid = f"ab_{args.condition}_{i:02d}_{stamp}"
        print(f"[{i + 1}/{args.limit}] sid={sid} :: {q[:60]}", flush=True)
        r = await run_one(q, sid, args.cap)
        print(f"    -> {r['duration_s']}s len={r['resp_len']} err={r['error']}", flush=True)
        runs.append(r)

    outdir = REPO / "scripts/experiments/results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"ab_{args.condition}_{datetime.date.today().isoformat()}_{stamp}.json"
    out.write_text(json.dumps(
        {"condition": args.condition, "model": os.environ["LLM__MAIN_MODEL"],
         "session_prefix": f"ab_{args.condition}_", "runs": runs},
        ensure_ascii=False, indent=1))
    ok = sum(1 for r in runs if not r["error"] and r["resp_len"] > 0)
    print(f"\n=== done: {ok}/{len(runs)} produced a response · saved {out} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
