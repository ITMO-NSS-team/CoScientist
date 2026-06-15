#!/usr/bin/env python
"""Deep offline analysis of a dumped Opik trace folder (no network).

Mines every trace+span for the failure modes DEVGRAPH claims to track, with a
temporal split (per day) so a "fixed on date X" claim can be checked against
whether the error still occurs AFTER X. Prints a structured report.

Usage: python scripts/opik_eval/analyze_dump.py [opik_dump/traces_since_2026-06-12]
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "opik_dump/traces_since_2026-06-12"

FAM_PREFIXES = ("ab_A_", "ab_B2_", "ab_B_", "ab_S3_", "ab_S4_", "ab_S5_", "ab_Snone_",
                "ab_probe", "ab_smoke", "l_L1_", "l_L2_", "l_Lcrit_", "l_Lplan_",
                "l_Lnone_", "session_001")
NOTFOUND_RE = re.compile(r"Tool '([^']+)' not found")
KEYERR_RE = re.compile(r"KeyError[:\s]+'?([A-Za-z_]\w*)'?")


def fam(tid: str) -> str:
    tid = str(tid or "")
    for p in FAM_PREFIXES:
        if tid.startswith(p):
            return p.rstrip("_")
    return "(adhoc/uuid)" if tid else "(none)"


def _err_fields(ei):
    """Normalize an error_info into (exception_type, message)."""
    if isinstance(ei, dict):
        return ei.get("exception_type") or ei.get("type") or "error", str(ei.get("message") or ei)
    return "error", str(ei)


def analyze():
    files = [f for f in glob.glob(f"{FOLDER}/*.json") if "index" not in f]
    traces = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        t = d["trace"]
        spans = d.get("spans", [])
        day = str(t.get("start_time") or "")[:10]
        rec = {
            "trace_id": t.get("id"), "thread_id": t.get("thread_id"),
            "fam": fam(t.get("thread_id")), "day": day,
            "model": (t.get("metadata") or {}).get("main_model", "?"),
            "dur": float(t["duration"]) / 1000 if t.get("duration") else 0.0,
            "n_llm": sum(1 for s in spans if s.get("type") == "llm"),
            "n_tool": sum(1 for s in spans if s.get("type") == "tool"),
            "trace_error": None, "exc": collections.Counter(),
            "notfound": collections.Counter(), "keyerr": collections.Counter(),
            "empties": 0, "tool_names": collections.Counter(),
            "reached_fedot": False, "reached_gen": False, "answer_len": 0,
            "credits": False, "get_state_exec": False, "msgs": [],
        }
        # trace-level outcome / answer length
        out = t.get("output")
        if isinstance(out, dict):
            rec["answer_len"] = len(json.dumps(out, default=str))
        if t.get("error_info"):
            et, msg = _err_fields(t["error_info"])
            rec["trace_error"] = et
            rec["exc"][et] += 1
            rec["msgs"].append((et, msg[:200]))
            _scan_msg(rec, et, msg)
        # spans
        for s in spans:
            nm = s.get("name")
            if s.get("type") == "tool":
                rec["tool_names"][nm] += 1
                if nm == "fedot_tool":
                    rec["reached_fedot"] = True
                if nm and "generate" in str(nm):
                    rec["reached_gen"] = True
            if str(nm or "").startswith("invoke_agent") or "molecule_generation" in str(nm or ""):
                rec["reached_gen"] = True
            if s.get("type") == "llm":
                o = s.get("output") if isinstance(s.get("output"), dict) else {}
                c = o.get("content")
                if c is None or (isinstance(c, str) and not c.strip()):
                    rec["empties"] += 1
            ei = s.get("error_info")
            if ei:
                et, msg = _err_fields(ei)
                rec["exc"][et] += 1
                rec["msgs"].append((et, msg[:200]))
                _scan_msg(rec, et, msg)
        traces.append(rec)
    return traces


def _scan_msg(rec, et, msg):
    for nm in NOTFOUND_RE.findall(msg):
        rec["notfound"][nm] += 1
    if et == "KeyError":
        for k in KEYERR_RE.findall("KeyError: " + msg) or KEYERR_RE.findall(msg):
            rec["keyerr"][k] += 1
    if "insufficient credits" in msg.lower():
        rec["credits"] = True
    if "get_state_from_server" in msg:
        rec["get_state_exec"] = True


def section(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main():
    T = analyze()
    n = len(T)
    section(f"CORPUS — {n} traces from {FOLDER}")
    by_day = collections.Counter(t["day"] for t in T)
    by_fam = collections.Counter(t["fam"] for t in T)
    by_model = collections.Counter(str(t["model"]).split("/")[-1] for t in T)
    print("by day  :", dict(sorted(by_day.items())))
    print("by model:", dict(by_model))
    print("by family (run condition):")
    for fm, c in by_fam.most_common():
        print(f"    {fm:18} {c}")

    section("OUTCOMES")
    errored = [t for t in T if t["exc"]]
    fedot = [t for t in T if t["reached_fedot"]]
    gen = [t for t in T if t["reached_gen"]]
    ans = [t for t in T if t["answer_len"] > 400]
    print(f"traces with >=1 error span/trace : {len(errored)}/{n} ({100*len(errored)//n}%)")
    print(f"reached fedot_tool               : {len(fedot)}/{n}")
    print(f"reached generation (gen/worker)  : {len(gen)}/{n}")
    print(f"produced a substantive answer    : {len(ans)}/{n} (output >400 chars)")
    # outcome by day
    for day in sorted(by_day):
        dd = [t for t in T if t["day"] == day]
        print(f"  {day}: errored {sum(1 for t in dd if t['exc'])}/{len(dd)} | "
              f"gen {sum(1 for t in dd if t['reached_gen'])}/{len(dd)} | "
              f"answer {sum(1 for t in dd if t['answer_len']>400)}/{len(dd)}")

    section("ERROR TAXONOMY (exception types) — total span+trace occurrences, and by day")
    allexc = collections.Counter()
    exc_by_day = collections.defaultdict(collections.Counter)
    for t in T:
        for et, c in t["exc"].items():
            allexc[et] += c
            exc_by_day[et][t["day"]] += c
    for et, c in allexc.most_common():
        traces_with = sum(1 for t in T if et in t["exc"])
        print(f"  {et:18} {c:4}  (in {traces_with} traces)  by-day {dict(sorted(exc_by_day[et].items()))}")

    section("SAMPLE MESSAGES per exception type")
    seen = set()
    for t in T:
        for et, msg in t["msgs"]:
            if et not in seen:
                seen.add(et)
                print(f"  [{et}] {msg}")

    section("TOOL-NOT-FOUND (hallucinated tool/agent names)")
    nf = collections.Counter()
    nf_fam = collections.defaultdict(set)
    for t in T:
        for nm, c in t["notfound"].items():
            nf[nm] += c
            nf_fam[nm].add(t["fam"])
    print(f"distinct hallucinated names: {len(nf)} | total occurrences: {sum(nf.values())}")
    for nm, c in nf.most_common():
        print(f"  {nm:32} x{c:3}  families={sorted(nf_fam[nm])}")

    section("KEY SIGNATURES vs DEVGRAPH CLAIMS")
    keyerr = collections.Counter()
    for t in T:
        keyerr.update(t["keyerr"])
    print(f"KeyError keys (F006.A3 fixed 'request'): {dict(keyerr)}")
    print(f"  KeyError traces by day: ",
          dict(sorted(collections.Counter(t['day'] for t in T if 'KeyError' in t['exc']).items())))
    eof = [t for t in T if "EOFError" in t["exc"]]
    print(f"EOFError (F001.A2 headless HITL): {len(eof)} traces, by day "
          f"{dict(sorted(collections.Counter(t['day'] for t in eof).items()))}")
    cred = [t for t in T if t["credits"]]
    print(f"APIError 'Insufficient credits' (NOT in DEVGRAPH?): {len(cred)} traces, by day "
          f"{dict(sorted(collections.Counter(t['day'] for t in cred).items()))}")
    gs = [t for t in T if t["get_state_exec"]]
    print(f"get_state_from_server execution (F000.A3 / B-crash): {len(gs)} traces, by day "
          f"{dict(sorted(collections.Counter(t['day'] for t in gs).items()))}")
    eg = [t for t in T if "ExceptionGroup" in t["exc"]]
    print(f"ExceptionGroup (FEDOT-internal, F000.A3 STAT3): {len(eg)} traces, by day "
          f"{dict(sorted(collections.Counter(t['day'] for t in eg).items()))}")

    section("RUNAWAY / OVER-EXPLORATION")
    runaway = [t for t in T if t["n_llm"] >= 25 or t["dur"] >= 690]
    print(f"runaway (n_llm>=25 or dur>=690s): {len(runaway)}/{n}")
    print(f"  median n_llm={_median([t['n_llm'] for t in T])} "
          f"median n_tool={_median([t['n_tool'] for t in T])} "
          f"median dur={_median([round(t['dur']) for t in T])}s")
    explore = collections.Counter()
    for t in T:
        for nm in ("search_mcp_servers", "retrieve_tools", "list_available_tools",
                   "list_server_tools", "get_server_info", "search_papers"):
            explore[nm] += t["tool_names"].get(nm, 0)
    print(f"  exploration-tool totals: {dict(explore)}")


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


if __name__ == "__main__":
    main()
