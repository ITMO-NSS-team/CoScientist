import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import CrossEncoder

model: CrossEncoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = CrossEncoder(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length=2048,
        trust_remote_code=True
    )
    model.predict([["warmup", "query"]])  # Прогрев модели
    logging.info("Reranker model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/rerank")
async def rerank(pairs: list[tuple[str, str]]):
    scores = model.predict(pairs).tolist()
    return {"scores": scores}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
