"""Generate a presigned URL for an object in the S3-compatible bucket.

Loads credentials from `automl/.env` (ENDPOINT_URL, ACCESS_KEY, SECRET_KEY,
BUCKET_NAME). Optionally lists matching objects by prefix.

Examples:
    # presign a single key (default bucket from .env, 1h expiry)
    python automl/scripts/s3_presign.py train/Alzheimer.csv

    # custom expiry (seconds) and explicit bucket
    python automl/scripts/s3_presign.py --bucket molecule-generative-mcp \\
        --expires 86400 train/Alzheimer.csv

    # search by substring inside an optional prefix, then presign each match
    python automl/scripts/s3_presign.py --prefix train/ --search Alzheimer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

DEFAULT_ENV =".env"# Path(__file__).resolve().parent.parent / 


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=(os.getenv("ENDPOINT_URL") or "").strip(),
        aws_access_key_id=(os.getenv("ACCESS_KEY") or "").strip(),
        aws_secret_access_key=(os.getenv("SECRET_KEY") or "").strip(),
        config=Config(signature_version="s3v4"),
    )


def list_keys(client, bucket: str, prefix: str, search: str | None) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    matches: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if search is None or search.lower() in key.lower():
                matches.append(key)
    return matches


def presign(client, bucket: str, key: str, expires: int) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", nargs="?", help="Exact S3 key to presign.")
    parser.add_argument("--bucket", default=None, help="Bucket name (default: BUCKET_NAME from .env).")
    parser.add_argument("--prefix", default="", help="Search prefix for --search mode.")
    parser.add_argument("--search", default=None, help="Case-insensitive substring filter on key names.")
    parser.add_argument("--expires", type=int, default=3600, help="Presigned URL TTL in seconds. Default 3600.")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help=f"Path to .env. Default: {DEFAULT_ENV}")
    args = parser.parse_args()

    load_dotenv(args.env)

    bucket = args.bucket or (os.getenv("BUCKET_NAME") or "").strip()
    if not bucket:
        print("ERROR: bucket not specified and BUCKET_NAME missing from .env", file=sys.stderr)
        return 2

    client = make_client()

    if args.search is not None or (args.key is None and args.prefix):
        keys = list_keys(client, bucket, args.prefix, args.search)
        if not keys:
            print(f"No keys found in s3://{bucket}/{args.prefix} matching {args.search!r}")
            return 1
        for key in keys:
            url = presign(client, bucket, key, args.expires)
            print(f"{key}\t{url}")
        return 0

    if not args.key:
        parser.error("either positional KEY or --search must be provided")

    url = presign(client, bucket, args.key, args.expires)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
