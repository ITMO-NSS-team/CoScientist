from pathlib import Path

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from CoScientist.papers_processing_refactoring.app.config_loader import get_settings
from CoScientist.papers_processing_refactoring.etl import *
from CoScientist.papers_processing_refactoring.embeddings import *
from CoScientist.papers_processing_refactoring.sources.local import LocalSource
from CoScientist.papers_processing_refactoring.storage.state.state_db import SQLiteStateManager
from CoScientist.papers_processing_refactoring.storage.artifacts import *
from CoScientist.papers_processing_refactoring.storage.vector import *
from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH

# from CoScientist.papers_processing_refactoring.retrieval import TwoStageRetriever

load_dotenv(CONFIG_PATH)
etl_settings = get_settings()


def build_services(settings):

    embedding_model = create_embedding_model({
        "type": settings.embeddings.type,
        "url": settings.embeddings.api_url,
        "model_name": settings.embeddings.model_name,
        "batch_size": settings.embeddings.batch_size,
    })
    
    return {
        "embedding_model": embedding_model
    }


def build_vector_store(settings):

    # if settings.vectordb.backend == "qdrant":
    #     return QdrantVectorStore(etl_settings.vectordb.qdrant)

    if settings.vectordb.backend == "chromadb":
        return ChromaVectorStore(
            settings.vectordb.chroma.host,
            settings.vectordb.chroma.port,
            settings.vectordb.chroma.collection
        )
    else:
        raise ValueError("Vector store configuration must be provided")
    

def build_artifacts_stores(settings):
    etl_art_store = S3ETLArtifactStore(
        endpoint=settings.s3.endpoint,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        bucket=settings.s3.etl_bucket
    )
    public_art_store = S3DomainArtifactStore(
        endpoint=settings.s3.endpoint,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        bucket=settings.s3.public_bucket  # Delete after testing
    )
    return etl_art_store, public_art_store


# TODO: implement SQL state storage connector initialization
# def build_state_store(settings):
#     pass


def main(settings):
    papers_dir = Path("/home/kamilfatkhiev/work_data/chem_projects/test_papers")
    # artifacts_dir = "/home/kamilfatkhiev/work_data/chem_projects/test_artifacts"
    # artifact_store = MockArtifactStore(artifacts_dir)
    
    source = LocalSource(papers_dir)
    vector_store = build_vector_store(settings)
    artifact_store, public_store = build_artifacts_stores(settings)
    llm_model = ChatOpenAI(
        model=settings.llm.llm_name,
        base_url=settings.llm.llm_base_url,
        api_key=settings.llm.llm_api_key
    )
    embedding_model = build_services(settings)["embedding_model"]

    with SQLiteStateManager(settings.database.sqlite_path) as state_manager:
        state_manager.reset_running_states()

        pipeline = ETLPipeline(
            steps=[
                FetchStep(source=source),
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

        print("Starting Pipeline...")
        for article in source.list_articles():
            print(f"\n--- Processing: {article.name} (ID: {article.id}) ---")

            ctx = ETLContext(
                article=article,
                state_manager=state_manager,
                artifact_store=artifact_store,
                public_store=public_store,
                vector_store=vector_store,
                llm=llm_model,
                embedding_model=embedding_model
            )

            try:
                pipeline.run(ctx)
                print("Article processing finished successfully.")
            except Exception as e:
                print(f"Article processing stopped due to error: {e}")
    
    # retriever = TwoStageRetriever(vector_store, embedding_model)
    # res = retriever.retrieve("Simple query", rerank_k=20, filters={"role": {"$eq": "body"}})
    # res_with_image = [r for r in res if eval(r.metadata["imgs_in_chunk"])]
    # for c in res_with_image:
    #     print(c)
    #
    # for r in res_with_image:
    #     for img_name in eval(r.metadata["imgs_in_chunk"]):
    #         public_store.download_image_from_s3(
    #             "",
    #             r.article_id,
    #             img_name,
    #             f"imgs/{r.article_id + "-" + img_name}"
    #         )


if __name__ == "__main__":
    main(etl_settings)
    
    # with SQLiteStateManager(etl_settings.database.sqlite_path) as state_manager:
    #     # state_manager.clear_data()
    #
    #     entries = state_manager.list_states(status="done")
    #     if entries:
    #         for entry in entries:
    #             print(entry)
    #     else:
    #         print("No records in StateDB")
    # vector_store = build_vector_store(etl_settings)
    # if "new_etl_test" in [col.name for col in vector_store.show_collections()]:
    #     vector_store.delete_collection("new_etl_test")
    #     print("Collection deleted")
    