"""
Tests for src/r2_storage.py

All boto3 calls are mocked — no real Cloudflare R2 connection is made.
"""

import io
import pytest
from unittest.mock import MagicMock, call
from botocore.exceptions import ClientError

from src.r2_storage import (
    R2Config,
    make_object_key,
    upload_fileobj,
    download_as_bytes,
    delete_object,
    list_uploads,
    generate_presigned_post,
    object_exists,
    get_object_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_error(code: str) -> ClientError:
    """Build a botocore ClientError with the given error code."""
    return ClientError(
        {"Error": {"Code": code, "Message": "test error"}},
        "HeadObject",
    )


def mock_client():
    return MagicMock()


# ---------------------------------------------------------------------------
# make_object_key
# ---------------------------------------------------------------------------

class TestMakeObjectKey:
    def test_format(self):
        key = make_object_key("Jane Smith", "report.pdf")
        assert key.startswith("uploads/Jane_Smith/")
        assert key.endswith("_report.pdf")

    def test_spaces_replaced_in_claimant(self):
        key = make_object_key("John Doe", "file.pdf")
        assert "John_Doe" in key
        assert " " not in key

    def test_empty_claimant_uses_unknown(self):
        key = make_object_key("", "report.pdf")
        assert key.startswith("uploads/unknown/")

    def test_whitespace_only_claimant_uses_unknown(self):
        key = make_object_key("   ", "report.pdf")
        assert key.startswith("uploads/unknown/")

    def test_uniqueness(self):
        k1 = make_object_key("Jane Smith", "report.pdf")
        k2 = make_object_key("Jane Smith", "report.pdf")
        assert k1 != k2


# ---------------------------------------------------------------------------
# generate_presigned_post
# ---------------------------------------------------------------------------

class TestGeneratePresignedPost:
    def test_returns_url_and_fields(self):
        client = mock_client()
        client.generate_presigned_post.return_value = {
            "url": "https://bucket.r2.cloudflarestorage.com/",
            "fields": {"key": "uploads/test.pdf", "policy": "base64policy"},
        }
        result = generate_presigned_post(client, "bucket", "uploads/test.pdf")
        assert "url" in result
        assert "fields" in result
        assert isinstance(result["url"], str)
        assert isinstance(result["fields"], dict)

    def test_calls_boto3_with_correct_bucket_and_key(self):
        client = mock_client()
        client.generate_presigned_post.return_value = {"url": "https://...", "fields": {}}
        generate_presigned_post(client, "mybucket", "uploads/mykey.pdf")
        kwargs = client.generate_presigned_post.call_args[1]
        assert kwargs["Bucket"] == "mybucket"
        assert kwargs["Key"] == "uploads/mykey.pdf"

    def test_respects_custom_expires_in(self):
        client = mock_client()
        client.generate_presigned_post.return_value = {"url": "https://...", "fields": {}}
        generate_presigned_post(client, "bucket", "key.pdf", expires_in=900)
        kwargs = client.generate_presigned_post.call_args[1]
        assert kwargs["ExpiresIn"] == 900

    def test_default_expires_in_is_one_hour(self):
        client = mock_client()
        client.generate_presigned_post.return_value = {"url": "https://...", "fields": {}}
        generate_presigned_post(client, "bucket", "key.pdf")
        kwargs = client.generate_presigned_post.call_args[1]
        assert kwargs["ExpiresIn"] == 3600

    def test_includes_pdf_content_type_condition(self):
        client = mock_client()
        client.generate_presigned_post.return_value = {"url": "https://...", "fields": {}}
        generate_presigned_post(client, "bucket", "key.pdf")
        kwargs = client.generate_presigned_post.call_args[1]
        assert kwargs["Fields"]["Content-Type"] == "application/pdf"


# ---------------------------------------------------------------------------
# object_exists
# ---------------------------------------------------------------------------

class TestObjectExists:
    def test_returns_true_when_object_exists(self):
        client = mock_client()
        assert object_exists(client, "bucket", "key") is True
        client.head_object.assert_called_once_with(Bucket="bucket", Key="key")

    def test_returns_false_on_404(self):
        client = mock_client()
        client.head_object.side_effect = _client_error("404")
        assert object_exists(client, "bucket", "missing") is False

    def test_returns_false_on_no_such_key(self):
        client = mock_client()
        client.head_object.side_effect = _client_error("NoSuchKey")
        assert object_exists(client, "bucket", "missing") is False

    def test_reraises_on_403(self):
        client = mock_client()
        client.head_object.side_effect = _client_error("403")
        with pytest.raises(ClientError):
            object_exists(client, "bucket", "key")

    def test_reraises_on_unexpected_error(self):
        client = mock_client()
        client.head_object.side_effect = _client_error("500")
        with pytest.raises(ClientError):
            object_exists(client, "bucket", "key")


# ---------------------------------------------------------------------------
# get_object_size
# ---------------------------------------------------------------------------

class TestGetObjectSize:
    def test_returns_content_length(self):
        client = mock_client()
        client.head_object.return_value = {"ContentLength": 1_048_576}
        assert get_object_size(client, "bucket", "key") == 1_048_576

    def test_calls_head_object_with_correct_args(self):
        client = mock_client()
        client.head_object.return_value = {"ContentLength": 100}
        get_object_size(client, "mybucket", "mykey")
        client.head_object.assert_called_once_with(Bucket="mybucket", Key="mykey")

    def test_zero_size(self):
        client = mock_client()
        client.head_object.return_value = {"ContentLength": 0}
        assert get_object_size(client, "bucket", "key") == 0


# ---------------------------------------------------------------------------
# upload_fileobj
# ---------------------------------------------------------------------------

class TestUploadFileobj:
    def test_calls_boto3_upload(self):
        client = mock_client()
        fileobj = io.BytesIO(b"pdf content")
        result = upload_fileobj(client, "bucket", fileobj, "uploads/key.pdf")
        client.upload_fileobj.assert_called_once_with(fileobj, "bucket", "uploads/key.pdf")

    def test_returns_object_key(self):
        client = mock_client()
        result = upload_fileobj(client, "bucket", io.BytesIO(b"data"), "uploads/key.pdf")
        assert result == "uploads/key.pdf"


# ---------------------------------------------------------------------------
# download_as_bytes
# ---------------------------------------------------------------------------

class TestDownloadAsBytes:
    def test_returns_bytes(self):
        client = mock_client()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: b"pdf bytes")}
        result = download_as_bytes(client, "bucket", "key")
        assert result == b"pdf bytes"

    def test_calls_get_object_with_correct_args(self):
        client = mock_client()
        client.get_object.return_value = {"Body": MagicMock(read=lambda: b"")}
        download_as_bytes(client, "mybucket", "mykey")
        client.get_object.assert_called_once_with(Bucket="mybucket", Key="mykey")


# ---------------------------------------------------------------------------
# delete_object
# ---------------------------------------------------------------------------

class TestDeleteObject:
    def test_calls_delete(self):
        client = mock_client()
        delete_object(client, "bucket", "uploads/key.pdf")
        client.delete_object.assert_called_once_with(Bucket="bucket", Key="uploads/key.pdf")


# ---------------------------------------------------------------------------
# list_uploads
# ---------------------------------------------------------------------------

class TestListUploads:
    def test_returns_list_of_dicts(self):
        from datetime import datetime
        client = mock_client()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "uploads/a.pdf", "Size": 100, "LastModified": datetime.now()},
                    {"Key": "uploads/b.pdf", "Size": 200, "LastModified": datetime.now()},
                ]
            }
        ]
        client.get_paginator.return_value = paginator
        result = list_uploads(client, "bucket")
        assert len(result) == 2
        assert result[0]["key"] == "uploads/a.pdf"
        assert result[0]["size"] == 100

    def test_empty_bucket_returns_empty_list(self):
        client = mock_client()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]  # no "Contents" key
        client.get_paginator.return_value = paginator
        result = list_uploads(client, "bucket")
        assert result == []
