import opik
from CoScientist.config import get_settings
s = get_settings()
c = opik.Opik(workspace="itmo-nss", api_key=s.opik.api_key, host="https://www.comet.com/opik/api/")

traces = c.search_traces(project_name="adk-coscientist", max_results=30)
print(f"{len(traces)} traces (newest first)\n", flush=True)
def short(x, n):
    x = "" if x is None else str(x).replace("\n"," ")
    return x[:n]
for t in traces:
    md = t.metadata or {}
    model = (md.get("main_model") or "?").split("/")[-1]
    err = short(t.error_info, 70)
    inp = ""
    try:
        inp = short(t.input.get("parts",[{}])[0].get("text",""), 55) if isinstance(t.input, dict) else short(t.input,55)
    except Exception:
        inp = short(t.input,55)
    dur = float(t.duration)/1000 if t.duration else 0
    print(f"{str(t.start_time)[:19]} | {model:14} | sp={t.span_count:>3} llm={t.llm_span_count:>2} tool={int(bool(t.has_tool_spans))} | {dur:6.1f}s | ${t.total_estimated_cost or 0:.4f} | err:{err:70} | {inp}", flush=True)
print("\nDONE", flush=True)
