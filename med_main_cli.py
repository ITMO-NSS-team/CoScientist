from typer.cli import state

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
import MedCoScientist.conf.create_conf as cc

import logging
from pathlib import Path
import sys
def setup_logger(name=__name__, log_level=logging.WARNING):
    """Setup a logger with consistent configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / 'app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger('my-app')
cc.conf['configurable']['logger'] = logger
# Paper analysis
graph = GraphBuilder(cc.conf)
# inputs = {"input": "question = 'How does the synthesis of Glionitrin A/B happen?'"}

# inputs = {"input": "What is the capital of France?"}
#inputs = {"input": "I would like to get PICO decomposition of the following hypothesis: Реперфузионное лечение у пациентов с тромбоэмболией легочной артерии высокого и промежуточного риска тридцатидневной летальности снижает риск развития посттромбоэмболического синдрома"}
inputs = {"input": "I would like to find relevant PubMed papers for the following hypothesis: Reperfusion therapy in patients with pulmonary embolism at high and intermediate risk of thirty-day mortality reduces the risk of developing post-thromboembolic syndrome."}

from pprint import pprint

if __name__ == "__main__":
    for step in graph.stream(inputs, user_id="1"):
        pprint(f"\n\nNEW STEP: \n{step}")
        # history = [f"{i[:30]}..." for i in step['past_steps']]
        # print(f"=====\n"
        #       f"PLAN: {step['plan']}\n"
        #       f"PAST_STEPS: {history}\n"
        #       f"NEXT_STEPS: {step['next']}\n"
        #       f"METADATA: {step['metadata']}\n"
        #       f"=====")
    pprint(f'\n\nFINAL RESULT\n {step['response']}')
    pprint(f'\n\nFINAL Papers\n {step['found_pubmed_papers']}')

