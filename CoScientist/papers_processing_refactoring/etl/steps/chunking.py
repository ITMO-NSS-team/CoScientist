from ..base import ETLStep
from ..context import ETLContext
from ...domain.entities import Chunk, ChunkRole, ImageInfo
from ...utils.html_processing.splitting import chunk_html_to_chunks, make_chunk_id


class ChunkingStep(ETLStep):
    
    name = "chunking"

    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        article_domain = ctx.article.domain
        
        html = ctx.artifact_store.get_html(article_id, "paper_summarisation")
        manifest_data = ctx.artifact_store.get_metadata(article_id, "paper_summarisation")
        summary_data = manifest_data["summary"]
        images_data = [ImageInfo(**data) for data in manifest_data["images"]]
        
        if not html:
            raise RuntimeError("ChunkingStep requires cleaned HTML")

        body_chunks = chunk_html_to_chunks(html, article_id, article_domain)

        image_caption_chunks = []
        for idx, img in enumerate(images_data):
            if not img.caption:
                continue

            text = img.caption.strip()
            if not text:
                continue

            chunk_id = make_chunk_id(
                article_id,
                ChunkRole.IMAGE_CAPTION,
                idx,
                text,
            )

            image_caption_chunks.append(
                Chunk(
                    id=chunk_id,
                    article_id=article_id,
                    domain=ctx.article.domain or "default",
                    role=ChunkRole.IMAGE_CAPTION.value,
                    modality="text",
                    content=text,
                    metadata={
                        "image_id": img.id
                    },
                )
            )
        
        summary_chunk = Chunk(
            id=make_chunk_id(article_id, ChunkRole.SUMMARY, 1, summary_data["paper_summary"]),
            article_id=article_id,
            domain=ctx.article.domain or "default",
            role=ChunkRole.SUMMARY.value,
            modality="text",
            content=summary_data["paper_summary"],
            metadata={
                "paper_title": summary_data["paper_title"],
                "publication_year": summary_data["publication_year"],
                "authors": summary_data["authors"],
                "source": summary_data["source"],
                "research_area": summary_data["research_area"]
            }
        )

        ctx.chunks = {
            ChunkRole.BODY: body_chunks,
            ChunkRole.IMAGE_CAPTION: image_caption_chunks,
            ChunkRole.SUMMARY: [summary_chunk]
        }
