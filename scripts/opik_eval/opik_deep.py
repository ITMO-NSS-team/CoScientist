import opik, json
from collections import Counter
from CoScientist.config import get_settings
s = get_settings()
c = opik.Opik(workspace="itmo-nss", api_key=s.opik.api_key, host="https://www.comet.com/opik/api/")

traces = c.search_traces(project_name="adk-coscientist", max_results=30)

def get(d, *path, default=None):
    for p in path:
        if isinstance(d, dict): d = d.get(p)
        else: return default
    return d if d is not None else default

def txt(x, n=120):
    return ("" if x is None else str(x).replace("\n"," "))[:n]

# select: every trace with an error + a few big runaway ones + the gpt-oss runs
sel = []
for t in traces:
    md = t.metadata or {}
    model = (md.get("main_model") or "?").split("/")[-1]
    big = (t.span_count or 0) >= 90
    if t.error_info or big or "gpt-oss" in model:
        sel.append(t)
sel = sel[:11]
print(f"deep-analyzing {len(sel)} traces\n", flush=True)

prov_probe_done = False
for t in sel:
    md = t.metadata or {}
    model = (md.get("main_model") or "?").split("/")[-1]
    spans = c.search_spans(project_name="adk-coscientist", trace_id=t.id, max_results=400)
    llm = [x for x in spans if getattr(x,"type",None)=="llm"]
    tool = [x for x in spans if getattr(x,"type",None)=="tool"]
    # empty llm outputs
    empty = 0
    finish = Counter()
    upstream = Counter()
    for x in llm:
        out = x.output if isinstance(x.output, dict) else {}
        content = out.get("content")
        if content is None or (isinstance(content,str) and content.strip()=="" ) or (isinstance(content,list) and not content):
            empty += 1
        finish[out.get("finish_reason")] += 1
        # try to find OpenRouter upstream provider
        cm = out.get("custom_metadata") or {}
        u = x.usage or {}
        cand = cm.get("provider") or cm.get("provider_name") or u.get("provider")
        if cand: upstream[cand]+=1
    # tool-not-found + loop detection
    notfound = []
    toolcalls = Counter()
    tool_errors = Counter()
    for x in tool:
        nm = getattr(x,"name",None)
        inp = x.input if isinstance(x.input,dict) else {}
        toolcalls[(nm, txt(json.dumps(inp,ensure_ascii=False,sort_keys=True),80))] += 1
        ei = getattr(x,"error_info",None)
        if ei: tool_errors[txt(ei,60)] += 1
    # span-level errors anywhere
    span_errs = Counter()
    for x in spans:
        ei = getattr(x,"error_info",None)
        if ei: span_errs[txt(ei,70)] += 1
    top_repeat = toolcalls.most_common(1)[0] if toolcalls else (("",""),0)
    dur = float(t.duration)/1000 if t.duration else 0
    print(f"── {str(t.start_time)[:19]} | {model} | sp={t.span_count} llm={len(llm)} tool={len(tool)} | {dur:.0f}s", flush=True)
    print(f"   query: {txt(get(t.input,'parts') and t.input['parts'][0].get('text') if isinstance(t.input,dict) else t.input, 90)}", flush=True)
    if t.error_info: print(f"   TRACE ERROR: {txt(t.error_info,150)}", flush=True)
    print(f"   empty_llm={empty}/{len(llm)}  finish={dict(finish)}", flush=True)
    if upstream: print(f"   upstream_provider={dict(upstream)}", flush=True)
    print(f"   max_repeat_toolcall={top_repeat[1]}x -> {top_repeat[0][0]} {top_repeat[0][1][:50]}", flush=True)
    if tool_errors: print(f"   tool_errors={dict(tool_errors)}", flush=True)
    # show non-llm/tool span errors (the ValueError tool-not-found etc surface at general/trace)
    interesting = {k:v for k,v in span_errs.items() if 'not found' in k.lower() or 'apierror' in k.lower() or 'keyerror' in k.lower() or 'timed out' in k.lower()}
    if interesting: print(f"   span_errs={interesting}", flush=True)
    # one-time: dump an llm output custom_metadata to locate upstream provider field
    if not prov_probe_done and llm:
        out = llm[0].output if isinstance(llm[0].output, dict) else {}
        print("   [probe] llm.output.custom_metadata =", txt(json.dumps(out.get('custom_metadata'),ensure_ascii=False),200), flush=True)
        print("   [probe] llm.usage keys w/ 'provider' =", [k for k in (llm[0].usage or {}) if 'provider' in k.lower()], flush=True)
        prov_probe_done = True
    print(flush=True)
print("DONE", flush=True)
