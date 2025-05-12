import logging
import os
import time
from pathlib import Path
import re
from typing import List, Any, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types import DoclingDocument
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import AnyUrl

from docling_core.experimental.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.experimental.serializer.common import create_ser_result
from docling_core.experimental.serializer.markdown import MarkdownPictureSerializer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer
from docling_core.types.doc.document import PictureDescriptionData
from typing_extensions import override

_log = logging.getLogger(__name__)
load_dotenv("../config.env")

IMAGE_RESOLUTION_SCALE = 2.0


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


if __name__ == "__main__":
    # parsed_doc = parsing_playground("/home/kamilfatkhiev/work_data/papers/10_1021_acs_joc_0c02350.pdf")
    # paper_doc_chunk(parsed_doc)
    # paper_doc_chunk(
    #     "10_1021_acs_joc_0c02350/"
    #     "10_1021_acs_joc_0c02350-with-images.json"
    # )
    
    documents = loader(parse_and_clean(
        "questions/jirát_et_al_2025_surface_defects_and_crystal_growth_of_apremilast.pdf"))
    for document in documents:
        print(document)
        
