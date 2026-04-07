from pathlib import Path
import tempfile

from ..base import ETLStep
from ..context import ETLContext
from ...sources.base import ArticleSource


class FetchStep(ETLStep):
    
    name = "fetching"

    def __init__(self, source: ArticleSource):
        self.source = source

    def run(self, ctx: ETLContext) -> None:
        raw_bytes = self.source.fetch(ctx.article)
        ctx.raw_data = raw_bytes

        tmp_dir = Path(tempfile.gettempdir()) / "papers_ingest"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = tmp_dir / f"{ctx.article.id}.pdf"
        pdf_path.write_bytes(raw_bytes)

        ctx.parsed_representation = pdf_path
