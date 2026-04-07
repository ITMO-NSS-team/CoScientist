from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class ETLArtifactStore(ABC):

    @abstractmethod
    def step_exists(self, article_id: str, step: str) -> bool:
        pass

    @abstractmethod
    def put_html(self, article_id: str, step: str, html: str) -> None:
        pass

    @abstractmethod
    def get_html(self, article_id: str, step: str) -> str:
        pass

    @abstractmethod
    def put_images(
        self,
        article_id: str,
        step: str,
        images: dict[str, Image.Image],
    ) -> None:
        pass

    @abstractmethod
    def list_images(self, article_id: str, step: str) -> list[str]:
        pass

    @abstractmethod
    def get_image(
        self,
        article_id: str,
        step: str,
        image_name: str,
    ) -> Image.Image:
        pass

    @abstractmethod
    def put_metadata(
        self,
        article_id: str,
        step: str,
        metadata: dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def get_metadata(
        self,
        article_id: str,
        step: str,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def delete_step(self, article_id: str, step: str) -> None:
        pass

    @abstractmethod
    def delete_article(self, article_id: str) -> None:
        pass


class DomainArtifactStore(ABC):

    @abstractmethod
    def publish_article(
        self,
        domain: str,
        article_id: str,
        html: str,
        images: dict[str, Image.Image],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def get_article_html(self, domain: str, article_id: str) -> str:
        pass

    @abstractmethod
    def get_image_url(
        self,
        domain: str,
        article_id: str,
        image_name: str,
    ) -> str:
        pass

    @abstractmethod
    def delete_article(self, domain: str, article_id: str) -> None:
        pass
