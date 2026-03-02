"""
Tests for src/report_analyzer.py

All Anthropic API calls are mocked — no real API calls are made.
The retry helper (_create_with_retry) is exercised indirectly through the
public functions; its retry logic is tested directly in TestCreateWithRetry.
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch

import anthropic
import httpx

from src.report_analyzer import (
    ReportBoundary,
    ReportSummary,
    _create_with_retry,
    detect_boundaries,
    classify_relevance,
    summarize_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(text: str):
    """Return a mock Anthropic client whose messages.create returns `text`."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


def _rate_limit_error():
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.RateLimitError("rate limit", response=response, body=None)


def _overload_error():
    response = httpx.Response(
        529,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.APIStatusError("overloaded", response=response, body=None)


def _boundary(start=1, end=5, title="Report", text=None):
    return ReportBoundary(
        start_page=start,
        end_page=end,
        title=title,
        text=text or ("report text content " * 30),
    )


# ---------------------------------------------------------------------------
# _create_with_retry
# ---------------------------------------------------------------------------

class TestCreateWithRetry:
    def test_returns_on_first_success(self):
        client = MagicMock()
        client.messages.create.return_value = "response"
        result = _create_with_retry(client, model="m", max_tokens=10, messages=[])
        assert result == "response"
        assert client.messages.create.call_count == 1

    def test_retries_on_rate_limit_error(self):
        client = MagicMock()
        client.messages.create.side_effect = [_rate_limit_error(), "ok"]
        with patch("src.report_analyzer.time.sleep"):
            result = _create_with_retry(client, model="m", max_tokens=10, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 2

    def test_retries_on_529_overload(self):
        client = MagicMock()
        client.messages.create.side_effect = [_overload_error(), "ok"]
        with patch("src.report_analyzer.time.sleep"):
            result = _create_with_retry(client, model="m", max_tokens=10, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 2

    def test_raises_after_max_retries(self):
        client = MagicMock()
        client.messages.create.side_effect = _rate_limit_error()
        with patch("src.report_analyzer.time.sleep"):
            with pytest.raises(anthropic.RateLimitError):
                _create_with_retry(client, model="m", max_tokens=10, messages=[])
        # 1 initial attempt + 5 retries = 6 total
        assert client.messages.create.call_count == 6

    def test_reraises_other_errors_immediately(self):
        client = MagicMock()
        client.messages.create.side_effect = ValueError("unexpected")
        with pytest.raises(ValueError):
            _create_with_retry(client, model="m", max_tokens=10, messages=[])
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# detect_boundaries
# ---------------------------------------------------------------------------

class TestDetectBoundaries:
    def test_parses_valid_json_array(self):
        payload = json.dumps([
            {"title": "MRI Report", "start_page": 1, "end_page": 3},
            {"title": "IME", "start_page": 4, "end_page": None},
        ])
        results = detect_boundaries(
            page_texts=["p1", "p2", "p3", "p4"],
            batch_start_page=1,
            client=_make_client(payload),
        )
        assert len(results) == 2
        assert results[0].title == "MRI Report"
        assert results[0].end_page == 3
        assert results[1].end_page is None

    def test_converts_relative_to_absolute_page_numbers(self):
        """Claude returns 1-based page numbers within the batch; pipeline needs absolute."""
        payload = json.dumps([{"title": "Report", "start_page": 1, "end_page": 2}])
        results = detect_boundaries(["p1", "p2"], batch_start_page=11, client=_make_client(payload))
        assert results[0].start_page == 11
        assert results[0].end_page == 12

    def test_strips_markdown_json_fence(self):
        payload = "```json\n" + json.dumps([{"title": "R", "start_page": 1, "end_page": 1}]) + "\n```"
        results = detect_boundaries(["text"], 1, _make_client(payload))
        assert len(results) == 1

    def test_strips_plain_code_fence(self):
        payload = "```\n" + json.dumps([{"title": "R", "start_page": 1, "end_page": 1}]) + "\n```"
        results = detect_boundaries(["text"], 1, _make_client(payload))
        assert len(results) == 1

    def test_falls_back_to_unknown_on_invalid_json(self):
        results = detect_boundaries(["text"], 1, _make_client("not valid json {{{"))
        assert len(results) == 1
        assert results[0].title == "Unknown Report"
        assert results[0].end_page is None

    def test_text_assembled_from_correct_pages(self):
        payload = json.dumps([{"title": "Report", "start_page": 1, "end_page": 2}])
        results = detect_boundaries(
            ["first page text", "second page text"], 1, _make_client(payload)
        )
        assert "first page text" in results[0].text
        assert "second page text" in results[0].text

    def test_null_end_page_gives_none(self):
        payload = json.dumps([{"title": "Open Report", "start_page": 1, "end_page": None}])
        results = detect_boundaries(["text"], 1, _make_client(payload))
        assert results[0].end_page is None

    def test_text_only_covers_report_pages(self):
        """A report on pages 2-3 should not include text from page 1."""
        payload = json.dumps([{"title": "Report", "start_page": 2, "end_page": 3}])
        results = detect_boundaries(
            ["page one", "page two", "page three"], batch_start_page=1, client=_make_client(payload)
        )
        assert "page one" not in results[0].text
        assert "page two" in results[0].text


# ---------------------------------------------------------------------------
# classify_relevance
# ---------------------------------------------------------------------------

class TestClassifyRelevance:
    def test_returns_true_when_relevant(self):
        client = _make_client('{"relevant": true, "reason": "IME report"}')
        assert classify_relevance(_boundary(), "Jane Smith", client) is True

    def test_returns_false_when_not_relevant(self):
        client = _make_client('{"relevant": false, "reason": "lab results"}')
        assert classify_relevance(_boundary(), "Jane Smith", client) is False

    def test_defaults_true_on_json_parse_failure(self):
        client = _make_client("not json at all")
        assert classify_relevance(_boundary(), "Jane Smith", client) is True

    def test_strips_markdown_fence_from_response(self):
        client = _make_client('```json\n{"relevant": false}\n```')
        assert classify_relevance(_boundary(), "Jane Smith", client) is False

    def test_only_sends_first_3000_chars(self):
        long_text = "x" * 10_000
        boundary = ReportBoundary(1, 5, "Report", long_text)
        client = _make_client('{"relevant": true}')
        classify_relevance(boundary, "Jane Smith", client)
        call_kwargs = client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        # The snippet is first 3000 chars; total message is larger but bounded
        assert len(user_content) < len(long_text)

    def test_always_uses_haiku_model(self):
        """The model parameter is ignored — Haiku is hardcoded for cost."""
        client = _make_client('{"relevant": true}')
        classify_relevance(_boundary(), "Jane Smith", client, model="claude-opus-4-6")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_empty_claimant_name_still_calls_api(self):
        client = _make_client('{"relevant": true}')
        result = classify_relevance(_boundary(), "", client)
        assert client.messages.create.called
        assert result is True


# ---------------------------------------------------------------------------
# summarize_report
# ---------------------------------------------------------------------------

class TestSummarizeReport:
    def test_returns_report_summary_instance(self):
        client = _make_client("Dr. Smith completed an IME.")
        result = summarize_report(_boundary(), "Summarize.", client)
        assert isinstance(result, ReportSummary)

    def test_summary_text_matches_response(self):
        client = _make_client("Dr. Smith completed an IME.")
        result = summarize_report(_boundary(), "Summarize.", client)
        assert result.summary == "Dr. Smith completed an IME."

    def test_title_from_boundary(self):
        client = _make_client("Summary.")
        b = _boundary(title="Orthopaedic IME")
        result = summarize_report(b, "Summarize.", client)
        assert result.title == "Orthopaedic IME"

    def test_page_numbers_from_boundary(self):
        client = _make_client("Summary.")
        b = _boundary(start=7, end=14)
        result = summarize_report(b, "Summarize.", client)
        assert result.start_page == 7
        assert result.end_page == 14

    def test_report_index_is_zero(self):
        """Pipeline sets report_index after calling summarize_report."""
        client = _make_client("Summary.")
        result = summarize_report(_boundary(), "Summarize.", client)
        assert result.report_index == 0

    def test_end_page_defaults_to_start_page_when_none(self):
        client = _make_client("Summary.")
        b = ReportBoundary(5, None, "Report", "text " * 50)
        result = summarize_report(b, "Summarize.", client)
        assert result.end_page == 5

    def test_truncates_long_text(self):
        client = _make_client("Summary.")
        b = ReportBoundary(1, 2, "Report", "x" * 100_000)
        summarize_report(b, "Summarize.", client, max_text_chars=500)
        call_kwargs = client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "[...text truncated due to length...]" in user_content

    def test_short_text_not_truncated(self):
        client = _make_client("Summary.")
        b = ReportBoundary(1, 2, "Report", "short text")
        summarize_report(b, "Summarize.", client)
        call_kwargs = client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "[...text truncated due to length...]" not in user_content

    def test_uses_configured_model(self):
        client = _make_client("Summary.")
        summarize_report(_boundary(), "Summarize.", client, model="claude-haiku-4-5-20251001")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
