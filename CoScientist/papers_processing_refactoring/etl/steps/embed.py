from ..base import ETLStep
from ..context import ETLContext


class EmbeddingStep(ETLStep):
    
    name = "embed"
    
    def run(self, ctx: ETLContext) -> None:
        if not ctx.chunks:
            raise RuntimeError("EmbeddingStep requires chunks")
        
        ctx.embeddings = {}
        
        for role, chunks in ctx.chunks.items():
            if not chunks:
                continue
            
            texts = [chunk.content for chunk in chunks]
            vectors = ctx.embedding_model.embed_documents(texts)
            
            ctx.embeddings[role] = {
                "chunk_ids": [chunk.id for chunk in chunks],
                "vectors": vectors,
            }
