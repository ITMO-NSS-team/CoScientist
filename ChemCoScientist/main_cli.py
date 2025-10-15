from typer.cli import state

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
import conf.create_conf as cc


# membrans
# inputs = {"input": "What are the three primary sources from which lithium is currently obtained?"}
# inputs = {"input": "What is the typical effect of modifying a nanofiltration membrane surface with a positively charged polymer layer (e.g., PEI) on its selectivity for Li+ over Mg2+, and what is the underlying principle?"}
inputs = {"input": "What is a key advantage of electrodialysis with bipolar membranes (EDBM) for lithium recovery, and how does the presence of competing monovalent ions like Na+ and K+ affect Li+ flux and energy consumption?"}

# nanozymes
# inputs = {"input": "What is the optimal pH for the catalytic activity of platinum-nickel nanoparticles, and what is the observed maximal reaction velocity (vmax) for these nanoparticles when catalyzing the oxidation of TMB?"}
# inputs = {"input": "How does the oxidase-like activity of nano-manganese dioxide (MnO2) change with increasing temperature from 10°C to 90°C, and what is the activity at 90°C?"}
# inputs = {"input": "What is the optimal pH value for the catalytic activity of Fe3O4@Apt composites, and how does this compare to bare Fe3O4NPs?"}

# analytical
# inputs = {"input": "How does the viscosity of a deep eutectic solvent (DES) composed of hexanoic acid and diphenylguanidine acetate (2:1 molar ratio) compare to a DES composed of hexanoic acid and diphenylguanidine (2:1 molar ratio) at 30 °C, and what are the approximate viscosity values for both?"}
# inputs = {"input": "What is the initial water content in a deep eutectic solvent (DES) composed of hexanoic acid and diphenylguanidine (1:1 molar ratio), and how does this change after contact with an aqueous phase?"}
# inputs = {"input": "When purifying antibiotics from a millet extract using solid phase extraction (SPE) with molecularly imprinted polymers (MIPs), what is the observed trend in the loss rates of antibiotics when comparing non-imprinted polymers (NIPs), conventional MIPs, and deep eutectic solvent (DES)-modified MIPs?"}

# polymer
# inputs = {"input": "For the polymerization of styrene using the catalyst Cp*2(Me)Zr(u-O)Ti(NMe2)3, how does the catalytic activity respond to an increase in the MAO-to-catalyst ratio, and what is the characteristic glass-transition temperature (Tg) of the resulting polymer?"}
# inputs = {"input": "What is the proposed rate-determining step for the oxidation of ethylene glycol to glycolic acid on an Au/NiO surface with oxygen vacancies, and what specific roles do the "AuNi alloy" and NiO-Ov structures at the interface play in this step?"}
# inputs = {"input": "What is the effect of using a lower molecular weight PEO (MW = 10,000 g/mol ) within the PI host on the performance of a Li/LiFePO₄ all-solid-state cell at a lower operating temperature of 30°C?"}

# Paper analysis
graph = GraphBuilder(cc.conf)
# inputs = {"input": "question = 'How does the synthesis of Glionitrin A/B happen?'"}

if __name__ == "__main__":
    for step in graph.stream(inputs, user_id="1"):
        print(f"=====\n"
              f"PLAN: {step['plan']}\n"
              f"PAST_STEPS: {[f"{i[:30]}..." for i in step['past_steps']]}\n"
              f"NEXT_STEPS: {step['next']}\n"
              f"METADATA: {step['metadata']}\n"
              f"=====")

