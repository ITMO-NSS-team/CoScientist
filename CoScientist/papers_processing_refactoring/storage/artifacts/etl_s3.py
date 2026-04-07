import json
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from PIL import Image


class S3ETLArtifactStore:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
    ):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

    @staticmethod
    def _prefix(article_id: str, step: str) -> str:
        return f"articles/{article_id}/{step}/"

    def step_exists(self, article_id: str, step: str) -> bool:
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self._prefix(article_id, step),
            MaxKeys=1,
        )
        return "Contents" in resp

    # ---------- HTML ----------

    def put_html(self, article_id: str, step: str, html: str) -> None:
        key = self._prefix(article_id, step) + "document.html"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html",
        )

    def get_html(self, article_id: str, step: str) -> str:
        key = self._prefix(article_id, step) + "document.html"
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    # ---------- Images ----------

    def put_images(
        self,
        article_id: str,
        step: str,
        images: dict[str, Image.Image],
    ) -> None:
        for name, img in images.items():
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            key = self._prefix(article_id, step) + f"images/{name}"
            self.client.upload_fileobj(buf, self.bucket, key)

    def list_images(self, article_id: str, step: str) -> list[str]:
        prefix = self._prefix(article_id, step) + "images/"
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return []
        return [Path(o["Key"]).name for o in resp["Contents"]]

    def get_image(
        self,
        article_id: str,
        step: str,
        image_name: str,
    ) -> Image.Image:
        key = self._prefix(article_id, step) + f"images/{image_name}"
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return Image.open(BytesIO(obj["Body"].read()))

    # ---------- Metadata ----------

    def put_metadata(
        self,
        article_id: str,
        step: str,
        metadata: dict[str, Any],
    ) -> None:
        key = self._prefix(article_id, step) + "meta.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metadata).encode("utf-8"),
            ContentType="application/json",
        )

    def get_metadata(self, article_id: str, step: str) -> dict[str, Any]:
        key = self._prefix(article_id, step) + "meta.json"
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    # ---------- Delete ----------

    def delete_step(self, article_id: str, step: str) -> None:
        prefix = self._prefix(article_id, step)
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return
        for obj in resp["Contents"]:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=obj["Key"],
            )

    def delete_article(self, article_id: str) -> None:
        prefix = f"articles/{article_id}/"
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return
        for obj in resp["Contents"]:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=obj["Key"],
            )
    
    def delete_file(self, article_id: str, step: str, filename: str) -> None:
        key = self._prefix(article_id, step) + filename
        self.client.delete_object(
            Bucket=self.bucket,
            Key=key,
        )


# for testing
class MockArtifactStore:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.root.mkdir(exist_ok=True)
    
    # ---------- HTML ----------
        
    def put_html(self, article_id: str, step: str, html: str) -> None:
        article_step_dir = Path(self.root, article_id, step)
        article_step_dir.mkdir(parents=True, exist_ok=True)
        path = article_step_dir / "document.html"
        mode = "wb" if isinstance(html, bytes) else "w"
        with open(path, mode) as f:
            f.write(html)
        
    def get_html(self, article_id, step):
        article_step_dir = Path(self.root, article_id, step)
        path = article_step_dir / "document.html"
        if not path.exists(): return None
        try:
            with open(path, "rb") as f:
                return f.read().decode('utf-8')
        except:
            return None
    
    # ---------- Images ----------
    
    def put_images(
            self,
            article_id: str,
            step: str,
            images: dict[str, Image.Image],
    ) -> None:
        article_step_dir = Path(self.root, article_id, step)
        images_dir = article_step_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for name, img in images.items():
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            path = images_dir / name
            with open(path, "wb") as f:
                f.write(buf.getvalue())
    
    def list_images(self, article_id: str, step: str) -> list[str]:
        article_step_dir = Path(self.root, article_id, step)
        images_dir = article_step_dir / "images"
        if not images_dir.exists():
            return []
        return [f.name for f in images_dir.iterdir() if f.is_file()]
    
    def get_image(
            self,
            article_id: str,
            step: str,
            image_name: str,
    ) -> Image.Image | None:
        article_step_dir = Path(self.root, article_id, step)
        path = article_step_dir / "images" / image_name
        if not path.exists():
            return None
        return Image.open(path)
    
    # ---------- Metadata ----------
    
    def put_metadata(
            self,
            article_id: str,
            step: str,
            metadata: dict[str, Any],
    ) -> None:
        article_step_dir = Path(self.root, article_id, step)
        article_step_dir.mkdir(parents=True, exist_ok=True)
        path = article_step_dir / "meta.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
    
    def get_metadata(self, article_id: str, step: str) -> dict[str, Any] | None:
        article_step_dir = Path(self.root, article_id, step)
        path = article_step_dir / "meta.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # ---------- Delete ----------
    
    @staticmethod
    def _delete_path(step_path: Path) -> None:
        if step_path.exists():
            from shutil import rmtree
            rmtree(step_path)

    def delete_step(self, article_id: str, step: str) -> None:
        article_step_dir = Path(self.root, article_id, step)
        self._delete_path(article_step_dir)
        
    def delete_file(self, article_id: str, step: str, filename: str) -> None:
        article_step_dir = Path(self.root, article_id, step)
        path = article_step_dir / filename
        if path.exists():
            path.unlink()
