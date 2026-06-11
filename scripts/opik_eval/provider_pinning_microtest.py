import time, json
from collections import Counter
import litellm
from CoScientist.config import get_settings
s = get_settings()
litellm.api_key = s.llm.openai_api_key
litellm.suppress_debug_info = True

PIN4 = ["deepinfra", "groq", "together", "fireworks"]
PROMPT = [{"role": "user", "content": "Reply with exactly one word: READY."}]

def served_provider(resp):
    # OpenRouter returns the upstream provider; litellm surfaces it in different places
    for getter in (
        lambda r: r.get("provider"),
        lambda r: r.model_extra.get("provider") if getattr(r, "model_extra", None) else None,
        lambda r: (r._hidden_params or {}).get("additional_headers", {}).get("x-litellm-provider"),
    ):
        try:
            v = getter(resp)
            if v: return v
        except Exception:
            pass
    return None

def one_call(model, pin):
    kw = {"model": model, "messages": PROMPT, "num_retries": 2, "timeout": 60, "max_tokens": 64}
    if pin:
        kw["extra_body"] = {"provider": {"only": PIN4, "allow_fallbacks": True}}
    t0 = time.time()
    try:
        r = litellm.completion(**kw)
        dt = time.time() - t0
        msg = r.choices[0].message.content
        fr = r.choices[0].finish_reason
        empty = (msg is None) or (str(msg).strip() == "")
        ct = (r.usage.completion_tokens if r.usage else None)
        return {"ok": True, "empty": empty, "provider": served_provider(r), "finish": fr,
                "dt": round(dt, 1), "ctok": ct, "raw": r}
    except Exception as e:
        return {"ok": False, "empty": True, "provider": None, "finish": None,
                "dt": round(time.time() - t0, 1), "err": f"{type(e).__name__}: {str(e)[:90]}"}

CONDS = [
    ("A gpt-oss-120b PIN-ON ", "openrouter/openai/gpt-oss-120b", True),
    ("B gpt-oss-120b PIN-OFF", "openrouter/openai/gpt-oss-120b", False),
    ("C qwen3-235b   PIN-OFF", "openrouter/qwen/qwen3-235b-a22b-2507", False),
]
N = 5
probed = False
for label, model, pin in CONDS:
    provs, empties, errs, lat = Counter(), 0, [], []
    for i in range(N):
        res = one_call(model, pin)
        if not probed and res.get("ok"):
            r = res["raw"]
            print("[probe] response.provider field search:",
                  "model=", r.model, "| top-level provider=", r.get("provider"),
                  "| hidden=", list((r._hidden_params or {}).keys())[:6], flush=True)
            probed = True
        if res["ok"]:
            provs[res["provider"] or "unknown"] += 1
            if res["empty"]: empties += 1
            lat.append(res["dt"])
        else:
            errs.append(res["err"])
        print(f"   {label} #{i+1}: ok={res['ok']} empty={res['empty']} prov={res.get('provider')} fin={res.get('finish')} {res['dt']}s ctok={res.get('ctok')}" + (f" ERR={res.get('err')}" if not res['ok'] else ""), flush=True)
    avg = round(sum(lat)/len(lat), 1) if lat else None
    print(f"== {label}: ok={N-len(errs)}/{N} empty={empties} providers={dict(provs)} errs={errs} avg_lat={avg}s\n", flush=True)
print("DONE", flush=True)
