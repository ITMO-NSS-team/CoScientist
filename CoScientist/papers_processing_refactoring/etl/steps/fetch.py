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
        ctx.artifact_store.put_file(ctx.article.id, self.name, "source.pdf", raw_bytes)
