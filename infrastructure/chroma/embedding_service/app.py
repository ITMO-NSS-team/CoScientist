import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

model: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    model.encode(["warmup"])
    logging.info("Embedding model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
async def embed(texts: list[str]):
    embeddings = model.encode(texts).tolist()
    return {"embeddings": embeddings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
