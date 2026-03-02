"""
Cloudflare R2 storage helpers (S3-compatible via boto3).

Used by the Streamlit UI to persist uploaded PDFs in R2 so that:
  - Large files survive server restarts and network interruptions
  - The pipeline can be re-run without re-uploading from the browser
  - No large temp files accumulate on server disk
"""

import uuid
from dataclasses import dataclass
from typing import IO, List, Dict, Any

import boto3
from botocore.config import Config


@dataclass
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str


def get_client(cfg: R2Config):
    """Return a boto3 S3 client pointed at the R2 endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=f"https://{cfg.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def make_object_key(claimant_name: str, filename: str) -> str:
    """Generate a unique R2 object key for an uploaded PDF."""
    uid = uuid.uuid4().hex[:8]
    safe = claimant_name.strip().replace(" ", "_") or "unknown"
    return f"uploads/{safe}/{uid}_{filename}"


def upload_fileobj(client, bucket: str, fileobj: IO[bytes], object_key: str) -> str:
    """
    Multipart upload from a file-like object to R2.

    boto3's upload_fileobj automatically uses multipart for large files,
    so this handles 500MB+ PDFs without exhausting memory.

    Returns the object_key on success.
    """
    client.upload_fileobj(fileobj, bucket, object_key)
    return object_key


def download_as_bytes(client, bucket: str, object_key: str) -> bytes:
    """Download an R2 object and return its full content as bytes."""
    resp = client.get_object(Bucket=bucket, Key=object_key)
    return resp["Body"].read()


def delete_object(client, bucket: str, object_key: str) -> None:
    """Permanently delete an object from R2."""
    client.delete_object(Bucket=bucket, Key=object_key)


def list_uploads(client, bucket: str, prefix: str = "uploads/") -> List[Dict[str, Any]]:
    """
    List all objects under the given prefix.

    Returns a list of dicts with keys: key, size, last_modified.
    """
    paginator = client.get_paginator("list_objects_v2")
    items: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            items.append(
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                }
            )
    return items
