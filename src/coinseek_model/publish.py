"""Publish versioned model files to a private S3-compatible store."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3


def publish(directory: Path, job_id: str) -> str:
    endpoint = os.environ["MODEL_S3_ENDPOINT"]
    bucket = os.environ["MODEL_S3_BUCKET"]
    prefix = os.getenv("MODEL_S3_PREFIX", "coinseek-models").strip("/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["MODEL_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MODEL_S3_SECRET_KEY"],
        region_name=os.getenv("MODEL_S3_REGION", "us-east-1"),
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model_id"] != job_id:
        raise ValueError("manifest model_id does not match workflow job_id")
    for filename, content_type in (
        ("model.onnx", "application/octet-stream"),
        ("manifest.json", "application/json"),
    ):
        key = f"{prefix}/{job_id}/{filename}"
        client.upload_file(
            str(directory / filename),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    return f"s3://{bucket}/{prefix}/{job_id}/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("model-output"))
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(publish(args.directory, args.job_id))


if __name__ == "__main__":
    main()
