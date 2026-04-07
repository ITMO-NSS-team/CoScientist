from typing import Iterable, List

from .base import ArticleSource


class SourceRegistry:

    def __init__(self) -> None:
        self._sources: List[ArticleSource] = []

    def register(self, source: ArticleSource) -> None:
        if source in self._sources:
            return
        self._sources.append(source)

    def list_sources(self) -> Iterable[ArticleSource]:
        return list(self._sources)
