from typing_extensions import TypedDict
from typing import Annotated, List, Tuple
import operator



class PlanExecute(TypedDict):
    """
    Plan and execute a sequence of steps.
    
        This class manages the planning and execution of tasks, maintaining a history of past steps
        and handling responses, visualizations, and potential language translation.
    """

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    next: str
    response: str
    visualization: str
    language: str
    translation: str
    automl_results: str
    nodes_calls: Annotated[List[Tuple], operator.add]
    #summary: Annotated[list[str], operator.add]