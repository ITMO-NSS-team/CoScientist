from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ArticleStatus(Enum):
    NEW = "new"
    PARSED = "parsed"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class ArticleProcessingState(BaseModel):
    article_id: str
    stage: ArticleStatus = ArticleStatus.NEW
    updated_at: datetime
    error: Optional[str]