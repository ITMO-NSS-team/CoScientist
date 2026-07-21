from ..base import ETLStep
from ..context import ETLContext


class PublishStep(ETLStep):
    
    name = "publish"
    
    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id
        
        manifest_data = ctx.artifact_store.get_metadata(article_id, "paper_summarisation")
        summary_data = manifest_data["summary"]
        
        chunks_to_upload = []
        vectors_to_upload = []
        
        if ctx.chunks and ctx.embeddings:
            for role, chunks in ctx.chunks.items():
                if role not in ctx.embeddings:
                    continue
                
                role_embeddings = ctx.embeddings[role]["vectors"]
                
                if len(chunks) == len(role_embeddings):
                    chunks_to_upload.extend(chunks)
                    vectors_to_upload.extend(role_embeddings)
        
        try:
            if chunks_to_upload:
                ctx.vector_store.delete_by_article_id(article_id)
                ctx.vector_store.upsert_chunks(chunks_to_upload, vectors_to_upload)
            
            pdf_data = ctx.artifact_store.get_file(article_id, "fetching", "source.pdf")
            html = ctx.artifact_store.get_html(article_id, "paper_summarisation")
            
            image_names = ctx.artifact_store.list_images(article_id, "image_captioning")
            images = {
                name: ctx.artifact_store.get_image(article_id, "image_captioning", name)
                for name in image_names
            }
            
            ctx.public_store.publish_article(
                domain=summary_data["domain"],
                article_id=article_id,
                paper_summary=summary_data["paper_summary"],
                html=html,
                images=images,
                metadata=manifest_data,
                pdf_data=pdf_data,
            )
            
            # Cleaning after successfully publication
            ctx.artifact_store.delete_article(article_id)
        
        except Exception as e:
            print(f"[{self.name}] Error publishing {article_id}. Rolling back vector store...")
            ctx.vector_store.delete_by_article_id(article_id)
            ctx.public_store.delete_article(summary_data["domain"], article_id)
            raise e
