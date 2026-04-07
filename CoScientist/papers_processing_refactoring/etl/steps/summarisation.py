from langchain_core.messages import HumanMessage

from ..base import ETLStep
from ..context import ETLContext
from ...utils.general_utils import ExpandedSummary
from ...utils.prompts import summarisation_prompt


class PaperSummarisatonStep(ETLStep):
    
    name = "paper_summarisation"
    
    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        
        html = ctx.artifact_store.get_html(article_id, "image_captioning")
        manifest_data = ctx.artifact_store.get_metadata(article_id, "image_captioning")
        
        # summary_llm = ctx.llm.with_structured_output(ExpandedSummary)
        # expanded_summary: ExpandedSummary = summary_llm.invoke([HumanMessage(content=summarisation_prompt + html)])
        
        # TODO: delete dummy data
        expanded_summary: ExpandedSummary = ExpandedSummary(
            paper_summary=f"Dummy summary of {article_id}",
            paper_title=f"Dummy title of {article_id}",
            publication_year=9999,
            authors="author_1, author_2",
            source="Dummy publisher",
            research_area="Dummy research area"
        )
        
        manifest_data["summary"] = {
            "paper_summary": expanded_summary.paper_summary,
            "paper_title": expanded_summary.paper_title,
            "publication_year": expanded_summary.publication_year,
            "authors": expanded_summary.authors,
            "source": expanded_summary.source,
            "research_area": expanded_summary.research_area
        }
        
        ctx.artifact_store.put_html(article_id, self.name, html)
        if manifest_data:
            ctx.artifact_store.put_metadata(article_id, self.name, manifest_data)
