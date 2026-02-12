from typer.cli import state

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
import MedCoScientist.conf.create_conf as cc


# inputs = {"input": "What is the capital of France?"}
# inputs = {"input": "I would like to get PICO decomposition of the following hypothesis: Реперфузионное лечение у пациентов с тромбоэмболией легочной артерии высокого и промежуточного риска тридцатидневной летальности снижает риск развития посттромбоэмболического синдрома"}
inputs = {"input": "I would like to find relevant PubMed papers for the following hypothesis: Reperfusion therapy in patients with pulmonary embolism at high and intermediate risk of thirty-day mortality reduces the risk of developing post-thrombotic syndrome."}

# Paper analysis
graph = GraphBuilder(cc.conf)
# inputs = {"input": "question = 'How does the synthesis of Glionitrin A/B happen?'"}

if __name__ == "__main__":
    for step in graph.stream(inputs, user_id="1"):
        history = [f"{i[:30]}..." for i in step['past_steps']]
        print(f"=====\n"
              f"PLAN: {step['plan']}\n"
              f"PAST_STEPS: {history}\n"
              f"NEXT_STEPS: {step['next']}\n"
              f"METADATA: {step['metadata']}\n"
              f"=====")

