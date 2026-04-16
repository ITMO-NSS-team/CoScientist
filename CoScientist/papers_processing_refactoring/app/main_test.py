from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import time
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from marker.models import create_model_dict

from CoScientist.papers_processing_refactoring.app.config_loader import get_settings
from CoScientist.papers_processing_refactoring.etl import *
from CoScientist.papers_processing_refactoring.embeddings import *
from CoScientist.papers_processing_refactoring.scheduling.scheduler import IngestionScheduler, Schedule
from CoScientist.papers_processing_refactoring.sources.local import LocalSource
from CoScientist.papers_processing_refactoring.storage.state.state_db import SQLiteStateManager
from CoScientist.papers_processing_refactoring.storage.artifacts import *
from CoScientist.papers_processing_refactoring.storage.vector import *
from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH

MAX_WORKERS = 3

load_dotenv(CONFIG_PATH)
settings = get_settings()


def build_services(etl_settings):
    embedding_model = create_embedding_model({
        "type": etl_settings.embeddings.type,
        "url": etl_settings.embeddings.api_url,
        "model_name": etl_settings.embeddings.model_name,
        "batch_size": etl_settings.embeddings.batch_size,
    })
    return {
        "embedding_model": embedding_model
    }


def build_vector_store(etl_settings):
    if etl_settings.vectordb.backend == "chromadb":
        return ChromaVectorStore(
            etl_settings.vectordb.chroma.host,
            etl_settings.vectordb.chroma.port,
            etl_settings.vectordb.chroma.collection
        )
    else:
        raise ValueError("Vector store configuration must be provided")
    

def build_artifacts_stores(etl_settings):
    etl_art_store = S3ETLArtifactStore(
        endpoint=etl_settings.s3.endpoint,
        access_key=etl_settings.s3.access_key,
        secret_key=etl_settings.s3.secret_key,
        bucket=etl_settings.s3.etl_bucket
    )
    public_art_store = S3DomainArtifactStore(
        endpoint=etl_settings.s3.endpoint,
        access_key=etl_settings.s3.access_key,
        secret_key=etl_settings.s3.secret_key,
        bucket=etl_settings.s3.public_bucket  # Delete after testing
    )
    return etl_art_store, public_art_store


def build_state_store(etl_settings):
    if etl_settings.database.type == "sqlite":
        return SQLiteStateManager(etl_settings.database.sqlite_path)
    # elif etl_settings.database.type == "postgres":
    #     return PostgreSQLStateManager(etl_settings.database.postgres.dsn)
    else:
        raise ValueError("State store configuration must be provided")


def process_single_article(article, app_settings, shared_models, parse_lock):
    print(f"[{article.name}] Thread started...")
    
    state_manager = build_state_store(app_settings)
    
    if state_manager.get_status(article.id, "publish") == "done":
        return f"[{article.name}] Already processed. Skipped."
    
    if any(elem["status"] == "running" for elem in state_manager.list_states(article.id)):
        return f"[{article.name}] Processing is already running. Skipped."
    
    local_source = LocalSource(settings.files.directory)
    vector_store = build_vector_store(settings)
    artifact_store, public_store = build_artifacts_stores(settings)
    llm_model = ChatOpenAI(
        model=settings.llm.llm_name,
        base_url=settings.llm.llm_base_url,
        api_key=settings.llm.llm_api_key,  # noqa
        temperature=0.1
    )
    embedding_model = build_services(settings)["embedding_model"]
    
    pipeline = ETLPipeline(
        steps=[
            FetchStep(source=local_source),
            ParseStep(shared_models=shared_models, parse_lock=parse_lock),
            HtmlCleaningStep(),
            ImageFilteringStep(),
            ImageCaptioningStep(),
            PaperSummarisatonStep(),
            ChunkingStep(),
            EmbeddingStep(),
            PublishStep()
        ]
    )
    
    ctx = ETLContext(
        article=article,
        state_manager=state_manager,
        artifact_store=artifact_store,
        public_store=public_store,
        vector_store=vector_store,
        llm=llm_model,
        embedding_model=embedding_model
    )
    
    start = time.perf_counter()
    
    try:
        pipeline.run(ctx)
        end = time.perf_counter()
        return f"[{article.id}] Success in {end - start:.2f}s"
    except Exception as e:
        end = time.perf_counter()
        return f"[{article.id}] Failed: {str(e)} in {end - start:.2f}s"


def handle_articles_batch(articles, shared_models, parse_lock):
    print(f"Scheduler found {len(articles)} articles. Starting parallel processing...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_article = {
            executor.submit(process_single_article, art, settings, shared_models, parse_lock): art
            for art in articles
        }
        
        for future in as_completed(future_to_article):
            result_msg = future.result()
            print(result_msg)


def main():
    print("Starting Papers ETL Daemon...")
    
    with build_state_store(settings) as state_manager:
        print("Cleaning up hanging tasks...")
        state_manager.reset_running_states()
    
    shared_parser_models = create_model_dict()
    parse_lock = threading.Lock()
    
    local_source = LocalSource(settings.files.directory)
    
    scheduler = IngestionScheduler(
        on_batch=lambda batch: handle_articles_batch(batch, shared_parser_models, parse_lock)
    )
    scheduler.register(local_source, Schedule(timedelta(minutes=1)))
    
    try:
        while True:
            scheduler.poll()
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down daemon...")


if __name__ == "__main__":
    main()
    