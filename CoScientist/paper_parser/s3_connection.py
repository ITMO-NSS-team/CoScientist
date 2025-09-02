import os
from io import BytesIO
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)


class S3BucketService:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "default",
    ) -> None:
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

    def create_s3_client(self) -> boto3.client:
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )
        return client

    def upload_file_object(
        self,
        prefix: str,
        source_file_name: str,
        file_path: str,
    ) -> None:
        client = self.create_s3_client()
        destination_path = str(Path(prefix, source_file_name))
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        buffer = BytesIO(content)
        client.upload_fileobj(buffer, self.bucket_name, destination_path)

    def list_objects(self, prefix: str) -> list[str]:
        client = self.create_s3_client()

        response = client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        storage_content: list[str] = []

        try:
            contents = response["Contents"]
        except KeyError:
            return storage_content

        for item in contents:
            storage_content.append(item["Key"])

        return storage_content

    def delete_file_object(self, prefix: str, source_file_name: str) -> None:
        client = self.create_s3_client()
        path_to_file = str(Path(prefix, source_file_name))
        client.delete_object(Bucket=self.bucket_name, Key=path_to_file)
        
    def create_new_bucket(self, bucket_name: str):
        client = self.create_s3_client()
        try:
            client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            print(e)
    
    def del_bucket(self, bucket_name: str):
        client = self.create_s3_client()
        try:
            client.delete_bucket(Bucket=bucket_name)
        except Exception as e:
            print(e)
    
    def generate_presigned_url(self, s3_key, method: str = 'get_object', expiration=360):
        client = self.create_s3_client()
        return client.generate_presigned_url(
            method,
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
    
    def download_image_from_s3(self, s3_key, local_path):
        client = self.create_s3_client()
        client.download_file(self.bucket_name, s3_key, local_path)
    
    def get_image_bytes_from_s3(self, s3_key, bucket_name):
        client = self.create_s3_client()
        response = client.get_object(Bucket=bucket_name, Key=s3_key)
        return response['Body'].read()


s3_service = S3BucketService(
        endpoint=os.getenv("ENDPOINT_URL"),
        access_key=os.getenv("ACCESS_KEY"),
        secret_key=os.getenv("SECRET_KEY"),
        bucket_name=os.getenv("BUCKET_NAME")
    )

if __name__ == "__main__":
    s3_client = s3_service.create_s3_client()
    # buckets = s3_client.list_buckets()
    # for bucket in buckets["Buckets"]:
    #     print(bucket["Name"], bucket["CreationDate"])
    # objects = s3_client.list_objects(Bucket="test")
    # for obj in objects["Contents"]:
    #     print(obj)
