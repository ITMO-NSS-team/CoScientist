from datetime import datetime, timedelta, timezone
from typing import Callable, Dict

from ..domain.entities import Article
from ..sources.base import ArticleSource


class Schedule:
    def __init__(self, interval: timedelta):
        self.interval = interval


class IngestionScheduler:

    def __init__(self, on_article: Callable[[Article], None]):
        self._on_article = on_article
        self._sources: Dict[ArticleSource, Schedule] = {}
        self._last_run: Dict[ArticleSource, datetime] = {}

    def register(self, source: ArticleSource, schedule: Schedule) -> None:
        self._sources[source] = schedule
        self._last_run.setdefault(source, datetime.min.replace(tzinfo=timezone.utc))

    def poll(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)

        for source, schedule in self._sources.items():
            last = self._last_run[source]

            if now - last < schedule.interval:
                continue

            for article in source.list_articles():
                self._on_article(article)

            self._last_run[source] = now
