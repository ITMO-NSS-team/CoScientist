import logging
import os
from pathlib import Path
import re
import time
from typing import List, Any, Optional

from bs4 import BeautifulSoup, Tag
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.experimental.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.experimental.serializer.common import create_ser_result
from docling_core.experimental.serializer.markdown import MarkdownPictureSerializer
from docling_core.types import DoclingDocument
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling_core.types.doc.document import PictureDescriptionData
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from protollm.connectors import create_llm_connector
from pydantic import AnyUrl
from typing_extensions import override

from answer_question import prompt_func, convert_to_base64

_log = logging.getLogger(__name__)
load_dotenv("../config.env")

IMAGE_RESOLUTION_SCALE = 2.0

cls_prompt = """
Act as a scientific image classifier. Analyze the input image from an academic paper and determine if it contains meaningful scientific information relevant to the article's content.

**Meaningful images (True):**
- Research diagrams, charts, or graphs
- Experimental photographs or microscopy images
- Data tables with substantive content
- Mathematical formulas/equations
- Technical schematics or flowcharts
- Biological/chemical structures
- Statistical visualizations

**Non-meaningful images (False):**
- Journal/publisher logos
- Decorative icons (social media, print, download etc.)
- Banner advertisements
- Author photos or institutional emblems
- Pure decorative elements
- Copyright watermarks
- Navigation buttons

**Output Rules:**
1. Return only single-word verdict: 'True' for meaningful images, 'False' for non-meaningful
2. Prioritize false negatives over false positives
3. Assume small size (<150px) suggests non-meaningful content
4. Ignore textual content within logos/icons

**Classification standard:**
An image is only 'True' if it conveys scientific data/results/methods essential for understanding the paper's research.

Now classify this image:"""

table_extraction_prompt = """
You are a scientific document analysis expert. Strictly follow these steps when processing the chemistry paper image:

1. Detect all table-like structures in the image:
   - If exactly ONE complete table exists (fully visible borders, headers, and data cells with no cropping/obscuring)
     → Extract tabular data and convert to HTML using <table>, <tr>, <th>, <td> tags
     → Return ONLY the HTML code

2. In ALL other cases return EXACTLY:
   'No table'
   This includes when:
   - No table is detected
   - Multiple tables exist
   - Table is incomplete/cropped/obscured
   - Table headers are missing
   - Part of table extends beyond image boundaries

3. Formatting rules:
   - Never add explanations, comments or markdown
   - Don't process non-tabular content
   - Skip text recognition outside tables
   - Omit confidence indicators
   - Output either pure HTML or exact string 'No table'

Critical: Your response must contain ONLY one of two options:
Option A: <table>...</table> (for single valid table)
Option B: No table (all other cases)
"""


def parse_and_clean(path, annotate_pic: bool = False):
    logging.basicConfig(level=logging.INFO)
    
    file_name = Path(path).stem
    output_dir = Path("./parse_results", file_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    if annotate_pic:  # Picture annotations
        pipeline_options.enable_remote_services = True
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = vlm_options("vis-google/gemini-2.0-flash-001")
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    start_time = time.time()
    conv_res = doc_converter.convert(path)
    
    table_counter = 0
    picture_counter = 0
    redundant_pics = []
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                    output_dir / f"{file_name}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
        
        if isinstance(element, PictureItem):
            img = element.get_image(conv_res.document)
            if (img.size[0] > 150) and (img.size[1] > 150):
                picture_counter += 1
                element_image_filename = (
                        output_dir / f"{file_name}-picture-{picture_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")
            else:
                redundant_pics.append(element.self_ref)
    
    conv_res.document = clean_up_doc(conv_res.document)
    if annotate_pic:
        conv_res.document = remove_redundant_desc(conv_res.document, redundant_pics)
    end_time = time.time() - start_time
    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")
    return conv_res.document


def loader(
        dl_doc, annotate_pic = False
) -> List[Document]:
    docs = []
    if annotate_pic:
        chunker = HybridChunker(
            serializer_provider=ImgAnnotationSerializerProvider(),
        )
    else:
        chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    for chunk in chunk_iter:
        if chunk.text.startswith("\n" * 5):
            continue
        docs.append(
            Document(
                page_content="passage: " + chunker.contextualize(chunk=chunk),
                metadata={
                    "source": dl_doc.origin.filename,
                    "dl_meta": chunk.meta.export_json_dict(),
                },
            )
        )
    
    return docs


def clean_up_doc(full_doc: str | DoclingDocument) -> DoclingDocument:
    blacklist = (r"author information|associated content|acknowledgments|references|data availability|"
                 r"declaration of competing interest|credit authorship contribution statement|funding|"
                 r"ethical statements|supplementary materials|conflict of interest|author contributions|"
                 r"data availability statement|ethics approval|supplementary information")
    appendix_index = None
    cut_index = None
    
    if isinstance(full_doc, str):
        doc = DoclingDocument.load_from_json(full_doc)
    else:
        doc = full_doc
    
    for text in doc.texts:
        if text.label == "section_header" and re.search(blacklist, text.text, re.IGNORECASE) and cut_index is None:
            cut_index = doc.texts.index(text)
        if text.label == "section_header" and ("appendix" in text.text.lower()):
            appendix_index = doc.texts.index(text)
            break
    
    all_chunks_length = len(doc.texts)
    if appendix_index:
        for i in range(cut_index, appendix_index + 1):
            doc.texts[i].orig = ""
            doc.texts[i].text = ""
    elif cut_index:
        for i in range(cut_index, all_chunks_length):
            doc.texts[i].orig = ""
            doc.texts[i].text = ""
    
    return doc


def remove_redundant_desc(full_doc: str | DoclingDocument, pics_to_clean: list[str]) -> DoclingDocument:
    if isinstance(full_doc, str):
        doc = DoclingDocument.load_from_json(full_doc)
    else:
        doc = full_doc
    
    for pic in doc.pictures:
        if pic.self_ref in pics_to_clean:
            for annotation in pic.annotations:
                if isinstance(annotation, PictureDescriptionData):
                    annotation.text = ""
    
    return doc


def parsing_playground(path, annotate_pic = False):
    logging.basicConfig(level=logging.INFO)
    
    input_doc_path = Path(path)
    file_name = Path(path).stem
    output_dir = Path(file_name)
    
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True
    )
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    if annotate_pic:  # Picture annotations
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = vlm_options("vis-google/gemini-2.0-flash-001")
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    start_time = time.time()
    
    conv_res = doc_converter.convert(input_doc_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                    output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
        
        if isinstance(element, PictureItem):
            img = element.get_image(conv_res.document)
            if (img.size[0] > 150) and (img.size[1] > 150):
                picture_counter += 1
                element_image_filename = (
                        output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")
    
    conv_res.document = clean_up_doc(conv_res.document)
    
    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)
    
    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
    
    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
    
    # Save JSON with embedded pictures
    json_filename = output_dir / f"{doc_filename}-with-images.json"
    conv_res.document.save_as_json(json_filename, image_mode=ImageRefMode.REFERENCED)
    
    end_time = time.time() - start_time
    
    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")
    
    return conv_res.document


def vlm_options(model: str):
    options = PictureDescriptionApiOptions(
        url=AnyUrl("https://api.vsegpt.ru/v1/chat/completions"),
        params=dict(
            model=model,
        ),
        headers={
            "Authorization": "Bearer " + os.getenv("VSE_GPT_KEY"),
        },
        prompt="Describe the image in three sentences. Be concise and accurate.",
        timeout=90,
    )
    return options


def paper_doc_chunk(doc: str | DoclingDocument):
    chunker = HybridChunker()
    output_lines = []
    
    if isinstance(doc, str):
        doc = DoclingDocument.load_from_json(doc)
    
    chunk_iter = chunker.chunk(dl_doc=doc)
    for i, chunk in enumerate(chunk_iter):
        print(f"=== {i} ===")
        print(f"chunk.text:\n{repr(f'{chunk.text[:300]}…')}")
        
        enriched_text = chunker.contextualize(chunk=chunk)
        print(f"chunker.serialize(chunk):\n{repr(f'{enriched_text[:300]}…')}")
        
        print(f"chunk's metadata:\n{chunk.meta}")
        
        print()
        
        output_lines.append(
            f"=== {i} ===\n"
            f"chunk.text:\n{repr(chunk.text)}\n"
            f"chunker.serialize(chunk):\n{repr(enriched_text)}\n"
            f"chunk's metadata:\n{chunk.meta}\n"
        )
    
    with open(f'chunks/{doc.name}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
        
def simple_conversion(path: str|Path):
    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)


class AnnotationPictureSerializer(MarkdownPictureSerializer):

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        
        res_parts: list[SerializationResult] = []
        
        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)
        
        for annotation in item.annotations:
            if isinstance(annotation, PictureDescriptionData):
                fig_desc = annotation.text if annotation.text else ""
                res_parts.append(create_ser_result(text=f"Figure description: {fig_desc}", span_source=item))
        
        text_res = "\n\n".join([r.text for r in res_parts])
        return create_ser_result(text=text_res, span_source=res_parts)
    

class ImgAnnotationSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=AnnotationPictureSerializer(),
        )
    

def parse_with_marker(paper_name: str, use_llm: bool=False) -> (str, Path):
    config = {
        "output_format": "html",
        "use_llm": use_llm,
        "openai_api_key": os.getenv("VSE_GPT_KEY"),
        "openai_model": "google/gemini-2.0-flash-lite-001",
        "openai_base_url": "https://api.vsegpt.ru/v1"
    }
    config_parser = ConfigParser(config)
    
    file_name = Path(paper_name)
    
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_parser.generate_config_dict(),
        renderer=config_parser.get_renderer(),
        llm_service="marker.services.openai.OpenAIService"
    )
    rendered = converter(paper_name)
    text, _, images = text_from_rendered(rendered)
    
    output_dir = Path("./parse_results", str(file_name.stem) + "_marker")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_output(rendered, output_dir=str(output_dir), fname_base=f"{file_name.stem}")
    return file_name.stem, output_dir


def clean_up_html(paper_name: str, doc_dir: Path) -> str:
    
    file_name = Path(paper_name + ".html")
    parsed_file_path = Path(doc_dir, file_name)
    with open(parsed_file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "lxml")
    
    blacklist = [
        "author information", "associated content", "acknowledgments", "acknowledgements", "references",
        "data availability", "declaration of competing interest", "credit authorship contribution statement", "funding",
        "ethical statements", "supplementary materials", "conflict of interest", "conflicts of interest",
        "author contributions", "data availability statement", "ethics approval", "supplementary information"
    ]
    for header in soup.find_all(["h1", "h2", "h3"]):
        header_text = header.get_text(strip=True).lower()
        
        if any(exclude in header_text for exclude in blacklist):
            next_node = header.next_sibling
            
            elements_to_remove = []
            while next_node and next_node.name not in ["h1", "h2"]:
                elements_to_remove.append(next_node)
                next_node = next_node.next_sibling
            
            header.decompose()
            for element in elements_to_remove:
                if isinstance(element, Tag):
                    element.decompose()
    
    llm = create_llm_connector("https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001")
    for img in soup.find_all('img'):
        img_path = str(doc_dir) + "/" + img.get("src")
        images = list(map(convert_to_base64, [img_path]))
        query = [prompt_func({"text": cls_prompt, "image": images})]
        res_1 = llm.invoke(query).content
        if res_1.strip() == "False":
            parent_p = img.find_parent('p')
            if parent_p:
                parent_p.decompose()
                os.remove(img_path)
        else:
            table_query = [prompt_func({"text": table_extraction_prompt, "image": images})]
            res_2 = llm.invoke(table_query).content
            if res_2 != "No table":
                pattern = r'<table\b[^>]*>.*?</table>'
                match = re.search(pattern, res_2, re.DOTALL)
                if match:
                    html_table = match.group(0)
                    table_soup = BeautifulSoup(html_table, 'html.parser')
                    parent_p = img.find_parent('p')
                    if parent_p:
                        parent_p.replace_with(table_soup)

    with open(Path(doc_dir, f"{file_name.stem}_processed.html"), "w", encoding='utf-8') as file:
        file.write(str(soup.prettify()))

    return soup.prettify()
    

def html_chunking(html_string: str, paper_name: str) -> list:
    
    def custom_table_extractor(table_tag):
        return str(table_tag).replace("\n", "")
    
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
    
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "],
        elements_to_preserve=["ul", "table", "ol"],
        preserve_images=True,
        custom_handlers={"table": custom_table_extractor}
    )
    
    documents = splitter.split_text(html_string)
    for doc in documents:
        doc.page_content = "passage: " + doc.page_content
        doc.metadata["imgs_in_chunk"] = str(extract_img_url(doc.page_content, paper_name))
        doc.metadata["source"] = paper_name + ".pdf"
        
    return documents


def extract_img_url(doc_text: str, p_name: str):
    pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    matches = re.findall(pattern, doc_text)
    return ["./parse_results/" + p_name + "_marker/" + entry[0] for entry in matches]


if __name__ == "__main__":
    # documents = loader(parse_and_clean(
    #     "papers/10_1021_acs_joc_0c02350.pdf"))
    # for document in documents:
    #     print(document)
    
    p_path = './papers'
    paper = "kowalska-et-al-2023-visible-light-promoted-3-2-cycloaddition-for-the-synthesis-of-cyclopenta-b-chromenocarbonitrile.pdf"
    paper_path = os.path.join(p_path, paper)
    f_name, dir_name = parse_with_marker(paper_name=paper_path)
    parsed_paper = clean_up_html(paper_name=f_name, doc_dir=dir_name)
    chunks = html_chunking(html_string=parsed_paper, paper_name=f_name)
    print(chunks)
