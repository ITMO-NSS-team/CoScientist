from typing import Sequence

from .base import ETLStep
from .context import ETLContext


class ETLPipeline:
    def __init__(self, steps: Sequence[ETLStep]):
        self.steps = steps

    def run(self, ctx: ETLContext) -> ETLContext | str:
        
        publish_status = ctx.state_manager.get_status(ctx.article.id, "publish")
        if publish_status == "done":
            print(f"Article {ctx.article.id} is already published.")
            return publish_status
        
        for step in self.steps:
            step.execute(ctx)
        return ctx
