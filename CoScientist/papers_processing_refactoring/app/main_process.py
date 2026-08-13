from __future__ import annotations

import logging
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def _bootstrap_coscientist_package() -> None:
    """Регистрирует облегчённую заглушку пакета CoScientist, чтобы импорт подпакетов не исполнял
    CoScientist/__init__.py в окружении обработки.

    Не трогает sys.path (отсутствуют коллизии имён со stdlib) и срабатывает
    исключительно тогда, когда CoScientist ещё не был импортирован.
    """
    if "CoScientist" in sys.modules:
        return

    # main_process.py -> app -> papers_processing_refactoring -> CoScientist
    coscientist_dir = Path(__file__).resolve().parents[2]

    package = types.ModuleType("CoScientist")
    package.__path__ = [str(coscientist_dir)]
    package.__package__ = "CoScientist"
    package.__file__ = str(coscientist_dir / "__init__.py")
    sys.modules["CoScientist"] = package


_bootstrap_coscientist_package()

from CoScientist.papers_processing_refactoring.app.config_loader import get_settings
from CoScientist.papers_processing_refactoring.etl import *
from CoScientist.papers_processing_refactoring.embeddings import *
from CoScientist.papers_processing_refactoring.scheduling.scheduler import IngestionScheduler, Schedule
from CoScientist.papers_processing_refactoring.sources.local import LocalSource
from CoScientist.papers_processing_refactoring.storage.state import *
from CoScientist.papers_processing_refactoring.storage.artifacts import *
from CoScientist.papers_processing_refactoring.storage.vector import *
from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_WORKERS = 3
SLEEP_TIME = 10
POOLING_TIME_MINUTES = 1

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
        secret_key=etl_settings.s3.secret_key.get_secret_value(),
        bucket=etl_settings.s3.etl_bucket
    )
    public_art_store = S3DomainArtifactStore(
        endpoint=etl_settings.s3.endpoint,
        access_key=etl_settings.s3.access_key,
        secret_key=etl_settings.s3.secret_key.get_secret_value(),
        bucket=etl_settings.s3.public_bucket
    )
    return etl_art_store, public_art_store


def build_state_store(etl_settings):
    if etl_settings.database.type == "sqlite":
        return SQLiteStateManager(etl_settings.database.sqlite_path)
    elif etl_settings.database.type == "postgres":
        return PostgreSQLStateManager(etl_settings.database.postgresql_dsn)
    else:
        raise ValueError("State store configuration must be provided")

# TODO: consider moving the initialisation of shared objects to the `main` function, taking thread safety into account
def process_single_article(article, app_settings):
    logger.info(f"[{article.name}] Thread started...")
    
    # TODO: wrap state manager in 'with' block
    state_manager = build_state_store(app_settings)
    
    if state_manager.get_status(article.id, "publish") == "done":
        return f"[{article.name}] Already processed. Skipped."
    
    if any(elem["status"] == "running" for elem in state_manager.list_states(article.id)):
        return f"[{article.name}] Processing is already running. Skipped."
    
    local_source = LocalSource(settings.files.directory)
    # TODO: wrap functions in 'try...except' block
    vector_store = build_vector_store(settings)
    artifact_store, public_store = build_artifacts_stores(settings)
    llm_model = ChatOpenAI(
        model=settings.llm.llm_name,
        base_url=settings.llm.llm_base_url,
        api_key=settings.llm.llm_api_key.get_secret_value(),  # noqa
        temperature=0.1
    )
    embedding_model = build_services(settings)["embedding_model"]
    
    pipeline = ETLPipeline(
        steps=[
            FetchStep(source=local_source),
            ParseStep(),
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
        logger.error(f"[{article.id}] Failed: {str(e)}", exc_info=True)
        return f"[{article.id}] Failed in {end - start:.2f}s"


def handle_articles_batch(articles):
    logger.info(f"Scheduler found {len(articles)} articles. Starting parallel processing...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_article = {
            executor.submit(process_single_article, art, settings): art
            for art in articles
        }
        
        # TODO: wrap in 'try...except' block
        for future in as_completed(future_to_article):
            result_msg = future.result()
            logger.info(result_msg)


def main():
    logger.info("Starting Papers ETL Daemon...")
    
    with build_state_store(settings) as state_manager:
        logger.info("Cleaning up hanging tasks...")
        state_manager.reset_running_states()
    
    local_source = LocalSource(settings.files.directory)
    
    scheduler = IngestionScheduler(
        on_batch=lambda batch: handle_articles_batch(batch)
    )
    scheduler.register(local_source, Schedule(timedelta(minutes=POOLING_TIME_MINUTES)))
    
    try:
        while True:
            scheduler.poll()
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        logger.info("Shutting down daemon...")


if __name__ == "__main__":
    main()
    