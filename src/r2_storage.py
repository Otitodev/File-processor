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


def generate_presigned_post(
    client,
    bucket: str,
    object_key: str,
    expires_in: int = 3600,
) -> dict:
    """
    Generate a presigned POST URL + form fields for a direct browser-to-R2 upload.

    The browser POSTs a multipart/form-data request directly to R2,
    bypassing any intermediate server (e.g. Railway's reverse proxy).

    Returns:
        dict with "url" (str) and "fields" (dict of form fields that must be
        submitted alongside the file, with "file" appended last per S3 spec).
    """
    return client.generate_presigned_post(
        Bucket=bucket,
        Key=object_key,
        Fields={"Content-Type": "application/pdf"},
        Conditions=[
            {"Content-Type": "application/pdf"},
            ["content-length-range", 1, 2 * 1024 * 1024 * 1024],  # 2 GB max
        ],
        ExpiresIn=expires_in,
    )


def object_exists(client, bucket: str, object_key: str) -> bool:
    """Return True if the given object exists in R2 (uses HeadObject, no data transfer)."""
    from botocore.exceptions import ClientError
    try:
        client.head_object(Bucket=bucket, Key=object_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def get_object_size(client, bucket: str, object_key: str) -> int:
    """Return the ContentLength in bytes of an existing R2 object."""
    return client.head_object(Bucket=bucket, Key=object_key)["ContentLength"]
