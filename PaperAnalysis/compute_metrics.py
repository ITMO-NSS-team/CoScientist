import datetime
import logging
from pathlib import Path

from chromadb.utils import embedding_functions
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
import pandas as pd

from answer_question import query_llm
from pipelines import retrieve_context

load_dotenv('../config.env')
from protollm.metrics import model_for_metrics

metrics_init_params = {
    "model": model_for_metrics,
    "verbose_mode": True,
    "async_mode": False,
}
# Можно переписать критерий для оценки. Текущий достаточно жестко оценивает
correctness_metric = GEval(
    name="Correctness",
    criteria=(
        "1. Correctness and Relevance:"
        "- Compare the actual response against the expected response. Determine the"
        " extent to which the actual response captures the key elements and concepts of"
        " the expected response."
        "- Assign higher scores to actual responses that accurately reflect the core"
        " information of the expected response, even if only partial."
        "2. Numerical Accuracy and Interpretation:"
        "- Pay particular attention to any numerical values present in the expected"
        " response. Verify that these values are correctly included in the actual"
        " response and accurately interpreted within the context."
        "- Ensure that units of measurement, scales, and numerical relationships are"
        " preserved and correctly conveyed."
        "3. Allowance for Partial Information:"
        "- Do not heavily penalize the actual response for incompleteness if it covers"
        " significant aspects of the expected response. Prioritize the correctness of"
        " provided information over total completeness."
        "4. Handling of Extraneous Information:"
        "- While additional information not present in the expected response should not"
        " necessarily reduce score, ensure that such additions do not introduce"
        " inaccuracies or deviate from the context of the expected response."
    ),
    evaluation_params=[
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
    def __init__(self):
        self.process_terminated = False
    
    def __enter__(self):
        self.start = datetime.datetime.now()
        return self
    
    @property
    def start_time(self):
        return self.start
    
    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start_time
    
    @property
    def seconds_from_start(self) -> float:
        return round(self.spent_time.total_seconds(), 2)
    
    def __exit__(self, *args):
        return self.process_terminated


def pipeline_test_with_save(
        data: pd.DataFrame,
        metrics_to_calculate: list,
        m_name: str,
        m_url: str,
        sum_collection: str,
        txt_collection: str,
        img_collection: str,
        version: float,
        sum_num: int = 1,
        txt_num: int = 3,
        img_num: int = 2
) -> pd.DataFrame:
    """Tests pipeline.

    Args:
        data: questions, correct context/answer etc.
        metrics_to_calculate: list of metrics to be calculated
        m_name: string with model name
        m_url: string with model URL and name
        sum_collection: paper summary collection name
        txt_collection: papers chunks summary collection name
        img_collection: images from papers collection name
        sum_num: number of papers for question
        txt_num: number of text chunks for question
        img_num: number of images for question
        version: test version

    Returns: pandas DataFrame
    """
    print("Pipeline test is running...")
    out_dir = Path("./test_results")
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
                    txt_data, img_data = retrieve_context(
                        sum_collection, txt_collection, img_collection, sum_num, txt_num, img_num,
                        "query: " + question
                    )
                    row_data["context_retrieve_time"] = t.seconds_from_start
                except Exception as e:
                    print(f"Context retrieval failed: {str(e)}")
                    txt_context = ''
                txt_context = ''
                img_paths = set()
                for chunk in txt_data['documents'][0]:
                    txt_context += '\n\n' + chunk.replace("passage: ", "")
                for chunk_meta in txt_data['metadatas'][0]:
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
    from chroma_db_operations import store_images_in_chromadb_txt_format, get_or_create_chroma_collection
    from pipelines import prepare_db
    
    sum_collection_name = 'paper_summaries'
    txt_collection_name = 'text_context_img2txt'
    img_collection_name = 'image_context'
    sum_chunk_num = 1  # Под вопрос будет подбираться одна статья по ее суммаризации
    txt_chunk_num = 3  # Количество текстовых чанков
    img_chunk_num = 2  # Количество изображений
    papers_path = './papers'  # Папка со статьями
    path_to_data = "./questions/complex_questions_draft.csv"  # Здесь указать файл с вопросами
    all_questions = pd.read_csv(path_to_data)

    rag_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        normalize_embeddings=True
    )

    model_name = "gemini-2.0-flash-001"
    model_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'
    
    # При первом запуске нужно создать векторные коллекции с помощью следующего кода
    sum_col, txt_col, img_col = prepare_db(sum_collection_name, txt_collection_name, img_collection_name,
                                           rag_embedding_function, rag_embedding_function, rag_embedding_function,
                                           store_images_in_chromadb_txt_format, papers_path, model_url)
    
    # При втором, если, например, просто промпты поменяли,
    # можно просто использовать следующий код, а верхний закомментировать
    # sum_col = get_or_create_chroma_collection(sum_collection_name, rag_embedding_function)
    # txt_col = get_or_create_chroma_collection(txt_collection_name, rag_embedding_function)
    # img_col = get_or_create_chroma_collection(img_collection_name, rag_embedding_function)

    v = 0.1
    pipeline_test_with_save(
        all_questions, [correctness_metric], model_name, model_url, sum_col, txt_col, img_col, v
    )
