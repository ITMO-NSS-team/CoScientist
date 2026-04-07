from enum import Enum
from pathlib import Path
from typing import Any, Literal, Dict, Optional, Union, Mapping

from pydantic import BaseModel

# from CoScientist.papers.sources.base import ArticleSource


# TODO: maybe add separate classes for source types (as Enum) and source references
class Article(BaseModel):
    id: str
    source_type: Literal["local", "remote"]
    source_ref: Union[str, Path]
    name: str
    domain: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    # source: ArticleSource
    

class ChunkRole(str, Enum):
    BODY = "body"
    SUMMARY = "summary"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"


class Chunk(BaseModel):
    id: str
    article_id: str
    domain: Optional[str] = None
    modality: Literal["text", "image"]
    content: str
    metadata: Optional[Mapping[str, Any]] = None
    role: str
    

class KnowledgeDomain(BaseModel):
    name: str
    description: str
    

class ImageInfo(BaseModel):
    id: str
    file_name: str
    original_src: str | Any
    is_kept: bool = True
    caption: Optional[str] = None
    final_s3_url: Optional[str] = None
