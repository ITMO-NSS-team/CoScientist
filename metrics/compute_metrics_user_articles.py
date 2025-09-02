import argparse
import logging
import os
from pathlib import Path
import time
from typing import List

import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from definitions import CONFIG_PATH
from ChemCoScientist.paper_analysis.question_processing import simple_query_llm
from protollm.metrics import model_for_metrics

load_dotenv(CONFIG_PATH)

VISION_LLM_URL = os.environ["VISION_LLM_URL"]

correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "If all essential information from the expected output is present in the actual output, regardless "
        "of wording or structure, it is OK.",
        "Actual output does not necessarily have to match word for word with the expected output.",
        "If the numeric values don't match, it's not OK.",
        "**It is STRICTLY FORBIDDEN to lower the score for expanding the answer** if the main meaning and data are correct.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=model_for_metrics,
    async_mode=False
)

def parse_arguments() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file with test data.")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment identifier.")
    parser.add_argument("--papers_path", type=str, required=True, help="Path to directory containing paper PDFs.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for results.")
    return parser.parse_args()

def test_pipeline(
    data: pd.DataFrame,
    metrics_to_calculate: list,
    exp_id: str,
    papers_dir: Path,
    out_dir: Path
) -> pd.DataFrame:
    """Pipeline for evaluating the assistant's performance in answering questions based on a user-provided article.
    
    Data requirements:
    * 'pdf' (str): PDF filename as in papers_dir.
    * 'question' (str): The question text.
    * 'answer' (str): The expected correct answer.

    Args:
        data: questions, correct context/answer etc.
        metrics_to_calculate: list of metrics to be calculated
        model_name: string with model name
        exp_id: experiment id
        papers_dir: path to directory with paper pdfs
        out_dir: path to directory with results

    Returns: pandas DataFrame
    """
    print("Pipeline started...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = VISION_LLM_URL.split("/")[-1]
    path_to_df = out_dir / f"{model_name}_v{exp_id}.csv"
    
    try:
        for i, row in data.iterrows():
            print(f"Processing question {i + 1} out of {len(data)}")
            pdf_path = papers_dir / f"{row['pdf']}"
            question = row["question"]
            correct_answer = row["answer"]
            
            row_data = {
            "pdf": pdf_path,
            "question": question,
            "correct_answer": correct_answer,
            "model_answer": "",
            "answer_generation_time": ""
            }
            
            start_time = time.time()
            llm_res = simple_query_llm(VISION_LLM_URL, question, [str(pdf_path)])
            end_time = time.time()
            row_data["model_answer"] = llm_res['answer']
            row_data["answer_generation_time"] = end_time - start_time
            
            test_case = LLMTestCase(
                        input=question,
                        actual_output=llm_res,
                        expected_output=correct_answer
                    )
            
            for metric in metrics_to_calculate:
                try:
                    metric.measure(test_case)
                    row_data[f"{metric.__name__}_score"] = metric.score
                    row_data[f"{metric.__name__}_reason"] = metric.reason
                except Exception as e:
                    row_data[f"{metric.__name__}_score"] = -1
                    row_data[f"{metric.__name__}_reason"] = f"{type(e).__name__}: {str(e)}"
                    
                row_df = pd.DataFrame([row_data])
            
            with open(path_to_df, 'a', newline='', encoding='utf-8') as f:
                row_df.to_csv(f, header=f.tell() == 0, index=False)
    except Exception as e:
            print(f"Critical error processing question {i}: {str(e)}")
            if 'row_df' in locals():
                with open(path_to_df, 'a', newline='', encoding='utf-8') as f:
                    row_df.to_csv(f, header=f.tell() == 0, index=False)
            raise
    
    results = pd.read_csv(path_to_df)
    print(f"Pipeline finished!\nAverage GEval score: {results['Correctness (GEval)_score'].mean()}\nAverage answer generation time: {results.answer_generation_time.mean()}")
         
if __name__ == "__main__":
    args = parse_arguments()

    data = pd.read_csv(args.data_path)
    papers_dir = Path(args.papers_path)
    out_dir = Path(args.out_path)

    test_pipeline(data, [correctness_metric], args.exp_id, papers_dir, out_dir)
    
"""
Example usage:

poetry run python compute_metrics_user_articles.py
    --data_path ../ChemCoScientist/paper_analysis/questions_dataset_prepared.csv
    --exp_id 0.2
    --papers_path ../ChemCoScientist/paper_analysis/papers/
    --out_path ../ChemCoScientist/paper_analysis/test_results/
"""