import datetime
import logging
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from definitions import CONFIG_PATH
from dotenv import load_dotenv
import pandas as pd

from ChemCoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from ChemCoScientist.paper_analysis.question_processing import query_llm

load_dotenv(CONFIG_PATH)
from protollm.metrics import model_for_metrics  # Иначе модель из конфига нормально не загружалась

metrics_init_params = {
    "model": model_for_metrics,
    "verbose_mode": True,
    "async_mode": False,
}
# Можно переписать критерий для оценки. Текущий достаточно жестко оценивает
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
answer_relevancy = AnswerRelevancyMetric(**metrics_init_params)
faithfulness = FaithfulnessMetric(**metrics_init_params)
context_precision = ContextualPrecisionMetric(**metrics_init_params)
context_recall = ContextualRecallMetric(**metrics_init_params)
context_relevancy = ContextualRelevancyMetric(**metrics_init_params)

logging.basicConfig(level=logging.INFO)


class Timer:
    """
    A context manager for measuring the execution time of a code block.
    
        Attributes:
            start_time: Stores the time when the timer is started.
            spent_time: Stores the total time elapsed.
    """

    def __init__(self):
        """
        Initializes a new instance of the Timer class.
        
        Sets the initial state of the process_terminated flag to False, 
        signifying that any associated process is initially running.
        
        Args:
            self: The Timer instance being initialized.
        
        Returns:
            None
        """
        self.process_terminated = False
    
    def __enter__(self):
        """
        Enters the context and records the start time.
        
        This method is called when a `with` statement is entered. It captures the current time to accurately measure the duration of the block of code that follows.
        
        Args:
            self: The Timer instance.
        
        Returns:
            self: The Timer instance, enabling its use within the `with` statement.
        """
        self.start = datetime.datetime.now()
        return self
    
    @property
    def start_time(self):
        """
        Returns the process's start time.
        
                This property provides access to the time when the timed operation began. 
                It's useful for calculating durations and analyzing performance.
        
                Returns:
                    float: The start time of the process, represented as a timestamp.
        """
        return self.start
    
    @property
    def spent_time(self) -> datetime.timedelta:
        """
        Calculates the time elapsed since the timer was started.
        
        Args:
                self: The Timer instance.
        
        Returns:
                datetime.timedelta: The time elapsed since the timer was started.
        """
        return datetime.datetime.now() - self.start_time
    
    @property
    def seconds_from_start(self) -> float:
        """
        Calculates the total elapsed time in seconds since the timer started.
        
        Args:
            self: The instance of the Timer class.
        
        Returns:
            float: The total elapsed time in seconds, rounded to two decimal places.
        
        This method provides a precise measurement of the duration the timer has been active, 
        allowing for accurate tracking of task completion times or process durations. 
        It achieves this by accessing the internally stored `spent_time` and converting it to total seconds.
        """
        return round(self.spent_time.total_seconds(), 2)
    
    def __exit__(self, *args):
        """
        Exits the context manager.
        
        This method is called when the `with` statement block is exited, allowing for resource cleanup and state finalization. It signals whether the managed process has already terminated.
        
        Args:
            self: The instance of the Timer context manager.
            *args: Positional arguments representing exception information 
                   passed from the `with` statement's exception handling (if any).
        
        Returns:
            bool: `True` if the managed process had terminated before exiting the context, `False` otherwise.
        """
        return self.process_terminated


def pipeline_test_with_save(
        data: pd.DataFrame,
        metrics_to_calculate: list,
        m_name: str,
        m_url: str,
        version: float,
        out_dir: Path,
        paper_store: ChromaDBPaperStore
) -> pd.DataFrame:
    """
    Evaluates the pipeline's performance by processing a dataset of questions and comparing model responses to expected answers.
    
    Args:
        data: DataFrame containing questions, correct answers, and associated paper/context information.
        metrics_to_calculate: List of metric functions to assess the quality of the pipeline's responses.
        m_name: Name of the model being tested.
        m_url: URL or identifier for the model.
        version: Version number of the test run.
        out_dir: Directory to store the test results.
        paper_store: Interface for retrieving relevant scientific context from a database.
    
    Returns:
        pandas DataFrame: DataFrame containing the test results, including questions, responses, metrics scores, and timing information.
    """
    print("Pipeline test is running...")
    out_dir.mkdir(parents=True, exist_ok=True)
    path_to_results = Path(out_dir, f"pipeline_test_{m_name}_v{version}.txt")
    path_to_df = Path(out_dir, f"pipeline_test_{m_name}_v{version}.csv")
    
    columns = [
        "index", "question", "correct_paper", "correct_context", "txt_context_from_db", "img_context_from_db",
        "correct_answer", "answer_from_model", "context_retrieve_time", "answer_generation_time"
    ]
    for metric in metrics_to_calculate:
        columns.append(f"{metric.__name__}_score")
        columns.append(f"{metric.__name__}_reason")
    
    if path_to_df.exists():
        existing_df = pd.read_csv(path_to_df)
        clear_existing_df = existing_df.drop_duplicates(subset=["index"], keep=False)
        clear_existing_df.to_csv(path_to_df, index=False)
        processed_indices = clear_existing_df["index"].unique().tolist() if "index" in clear_existing_df.columns else []
        start_index = max(processed_indices) + 1 if processed_indices else 0
    else:
        existing_df = pd.DataFrame(columns=columns)
        existing_df.to_csv(path_to_df, index=False)
        start_index = 0
    
    for i, row in data.iterrows():
        if i < start_index:
            continue
        
        try:
            print(f"Processing question {i}")
            question = row["question"].replace('"', "'")
            correct_answer = row["correct_answer"]
            correct_context = "\n".join(
                [row["correct_txt_context"], row["correct_img_context"], row["correct_table_context"]]
            )
            
            row_data = {
                "index": i,
                "question": question,
                "correct_paper": row["paper_name"],
                "correct_context": correct_context,
                "txt_context_from_db": None,
                "img_context_from_db": None,
                "correct_answer": correct_answer,
                "answer_from_model": "",
                "context_retrieve_time": None,
                "answer_generation_time": None
            }
            
            for metric in metrics_to_calculate:
                row_data[f"{metric.__name__}_score"] = -1
                row_data[f"{metric.__name__}_reason"] = ""
            
            with Timer() as t:
                try:
                    txt_data, img_data = paper_store.retrieve_context(question)
                    row_data["context_retrieve_time"] = t.seconds_from_start
                except Exception as e:
                    print(f"Context retrieval failed: {str(e)}")
                    txt_context = ''
                txt_context = ''
                img_paths = set()
                for idx, chunk in enumerate(txt_data, start=1):
                    txt_context += f"{idx}. Metadata: " \
                                   + str(chunk[2]) + "\nChunk: " \
                                   + chunk[1].replace("passage: ", "") + '\n\n'
                for chunk_meta in [chunk[2] for chunk in txt_data]:
                    img_paths.update(eval(chunk_meta["imgs_in_chunk"]))
                for img in img_data['metadatas'][0]:
                    img_paths.add(img['image_path'])
                row_data["txt_context_from_db"] = txt_context
                row_data["img_context_from_db"] = img_paths
                
            with Timer() as t:
                try:
                    llm_res, _ = query_llm(m_url, question, txt_context, list(img_paths))
                    row_data["answer_from_model"] = llm_res
                except Exception as e:
                    print(f"Answer generation failed: {str(e)}")
                    llm_res = ""
                row_data["answer_generation_time"] = t.seconds_from_start

            test_case = LLMTestCase(
                input=question,
                actual_output=llm_res,
                expected_output=correct_answer,
                context=[correct_context],
                retrieval_context=[txt_context],
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
            with open(path_to_df, 'a', newline='') as f:
                row_df.to_csv(f, header=f.tell() == 0, index=False)
        
        except Exception as e:
            print(f"Critical error processing question {i}: {str(e)}")
            if 'row_df' in locals():
                with open(path_to_df, 'a', newline='') as f:
                    row_df.to_csv(f, header=f.tell() == 0, index=False)
            raise
        
    result_df = pd.read_csv(path_to_df)
    result_df["total_time"] = (
            result_df["context_retrieve_time"] + result_df["answer_generation_time"]
    )
    # Calculation of basic statistics for exec time and function selection
    avg_context_retrieve_time = result_df["context_retrieve_time"].mean().round(2)
    avg_ans_generation_time = result_df["answer_generation_time"].mean().round(2)
    avg_total_time = result_df["total_time"].mean().round(2)
    # Calculation of statistics for metrics
    metrics_score_columns = list(filter(lambda x: "score" in x, result_df.columns.tolist()))
    metrics_to_print = []
    for column in metrics_score_columns:
        result_df[column] = pd.to_numeric(result_df[column])
        avg_score = result_df[result_df[column] != -1][column].mean()
        failed_evaluations = result_df[result_df[column] == -1].shape[0]
        metrics_to_print.append(
            f"- Average {column} is {avg_score}. Number of unsuccessfully processed questions {failed_evaluations}"
        )
    short_metrics_result = "\n".join(metrics_to_print)
    
    to_print = f"""Average context retrieving time: {avg_context_retrieve_time}
Average answer generation time: {avg_ans_generation_time}
Average total time: {avg_total_time}
Short metrics results:
{short_metrics_result}"""
    
    with open(path_to_results, "w") as f:
        print(to_print, file=f)
    
    return result_df


if __name__ == "__main__":
    papers_path = '../PaperAnalysis/papers'  # Папка со статьями
    path_to_data = "../PaperAnalysis/questions/complex_questions_draft.csv"  # Здесь указать файл с вопросами
    out_dir = Path("../PaperAnalysis/test_results")
    all_questions = pd.read_csv(path_to_data)

    model_name = "gemini-2.0-flash-001"
    model_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

    paper_store = ChromaDBPaperStore()
    # При первом запуске нужно создать векторные коллекции с помощью следующего кода
    # paper_store.prepare_db(papers_path)

    v = 0.1
    pipeline_test_with_save(
        all_questions, [correctness_metric], model_name, model_url, v, out_dir, paper_store
    )
