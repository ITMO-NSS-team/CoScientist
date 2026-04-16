import os
from pathlib import Path
import tempfile
import threading

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered

from ..base import ETLStep
from ..context import ETLContext


class ParseStep(ETLStep):

    name = "parsing"
    
    def __init__(self, shared_models: dict, parse_lock: threading.Lock):
        self.shared_models = shared_models
        self.parse_lock = parse_lock
        self.config_parser = ConfigParser({"output_format": os.getenv("OUTPUT_FORMAT", "html")})
    
    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id
        
        pdf_data = ctx.artifact_store.get_file(article_id, "fetching", "source.pdf")
        if not pdf_data:
            raise RuntimeError(f"ParseStep: PDF data not found for {article_id}")
        pdf_path = Path(tempfile.gettempdir()) / "papers_ingest" / f"{article_id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_data)
        
        with self.parse_lock:
            print(f"[{ctx.article.id}] Parsing is running (Lock acquired)...")
            
            converter = PdfConverter(
                artifact_dict=self.shared_models,
                config=self.config_parser.generate_config_dict(),
                renderer=self.config_parser.get_renderer()
            )
            
            # Newer versions of marker-pdf can accept bytes along with the file path
            rendered = converter(str(pdf_path))
            
            text, _, images = text_from_rendered(rendered)
            ctx.artifact_store.put_html(article_id, self.name, text)
            ctx.artifact_store.put_images(article_id, self.name, images)
        
        print(f"[{ctx.article.id}] Parsing finished (Lock released).")
