# automl_runner.py (should run in Python 3.10)
import sys
import json
from pathlib import Path
from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference
from fedotllm.agents.automl import AutoMLAgent


def main():
    """
    Loads configuration and state from command line arguments, prepares the environment, and generates a report using an AutoML agent.
    
    Args:
      None
    
    Returns:
      str: The generated report from the AutoML agent.
    """
    config = json.loads(sys.argv[1])
    state = json.loads(sys.argv[2])

    dataset_dir_path = Path(config["user_data_dir"]).resolve()
    work_dir = Path(config["user_data_dir"]).resolve()
    input_text = state["input"] if state['language'] == 'English' else state['translation']

    
    dataset = Dataset.from_path(dataset_dir_path)
    inference = AIInference(model=config["model_name"], base_url=config["openai_api_base"], api_key=config["openai_api_key"])
    automl = AutoMLAgent(inference=inference, dataset=dataset).create_graph()
    
    response = automl.invoke({'description': input_text, 'work_dir': work_dir})
    
    return response['report'] 

if __name__ == "__main__":
    result = main()
    print(json.dumps({"response": result}))