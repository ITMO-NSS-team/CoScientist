# import os
#
# from marker.config.parser import ConfigParser
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered

from ..base import ETLStep
from ..context import ETLContext

# TODO: delete after testing
from CoScientist.papers_processing_refactoring.utils.general_utils import generate_mock_images, generate_dummy_html


class ParseStep(ETLStep):

    name = "parsing"

    def __init__(self):
        # config_parser = ConfigParser({"output_format": os.getenv("OUTPUT_FORMAT", "html")})
        # self._converter = PdfConverter(
        #     artifact_dict=create_model_dict(),
        #     config=config_parser.generate_config_dict(),
        #     renderer=config_parser.get_renderer()
        # )
        pass

    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id
        
        pdf_path = ctx.parsed_representation
        if pdf_path is None:
            raise RuntimeError("ParseStep requires pdf Path in ctx.parsed_representation")

        # rendered = self._converter(str(pdf_path))
        # text, _, images = text_from_rendered(rendered)
        # ctx.artifact_store.put_html(article_id, self.name, text)
        # ctx.artifact_store.put_images(article_id, self.name, images)
        
        text = generate_dummy_html(article_id)
        images = generate_mock_images()
        ctx.artifact_store.put_html(article_id, self.name, text)
        ctx.artifact_store.put_images(article_id, self.name, images)
