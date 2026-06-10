"""MedicalAgent — PubMed search, PICO, study taxonomy, DICOM analysis."""
from google.adk.agents.llm_agent import LlmAgent

from CoScientist.agents.common import make_llm
from CoScientist.agents.callbacks import med_agent_before_model
from CoScientist.agents.prompts import medical_instruction
from CoScientist.tools import med_toolset_instance

medical_agent = LlmAgent(
    name="MedicalAgent",
    model=make_llm(),
    instruction=medical_instruction,
    description=(
        "Agent for medical and clinical questions: PubMed literature search, "
        "PICO extraction, study taxonomy, and DICOM image analysis."
    ),
    output_key="medical_results",
    tools=med_toolset_instance,
    before_model_callback=med_agent_before_model,
)

__all__ = ["medical_agent"]
