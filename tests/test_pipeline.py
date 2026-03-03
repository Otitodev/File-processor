"""
Tests for src/pipeline.py

All external dependencies (PDF extraction, OCR, Claude API) are mocked
so tests run fast with no network calls and no real PDFs needed.
"""

import dataclasses
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from src.pipeline import process_pdf
from src.report_analyzer import ReportBoundary, ReportSummary


# ---------------------------------------------------------------------------
# Module-level test fixtures
# ---------------------------------------------------------------------------

_DUMMY_BOUNDARY = ReportBoundary(
    start_page=1,
    end_page=5,
    title="IME Report",
    text="Patient examination and clinical findings text. " * 10,  # > 50 chars
)

_DUMMY_SUMMARY = ReportSummary(
    report_index=0,  # pipeline sets the real index
    title="IME Report",
    start_page=1,
    end_page=5,
    summary="Dr. Smith conducted an independent medical examination.",
)


def _fresh_summary(**overrides):
    """Return a fresh ReportSummary copy (avoids mutation between tests)."""
    return dataclasses.replace(_DUMMY_SUMMARY, **overrides)


@pytest.fixture
def mocks():
    """
    Patch every external dependency of process_pdf and yield a dict of mocks.
    Each test gets its own mock instances.
    """
    with (
        patch("src.pipeline.anthropic.Anthropic") as m_anthropic,
        patch("src.pipeline.get_page_count", return_value=5) as m_count,
        patch("src.pipeline.iter_page_batches") as m_batches,
        patch("src.pipeline.ocr_batch", return_value=["page text"] * 5) as m_ocr,
        patch("src.pipeline.detect_boundaries", return_value=[_DUMMY_BOUNDARY]) as m_detect,
        patch("src.pipeline.classify_relevance", return_value=True) as m_classify,
        patch("src.pipeline.summarize_report", side_effect=lambda **kw: _fresh_summary()) as m_summarize,
    ):
        m_batches.return_value = iter([(1, 5, [MagicMock()] * 5)])
        yield {
            "anthropic": m_anthropic,
            "get_page_count": m_count,
            "iter_page_batches": m_batches,
            "ocr_batch": m_ocr,
            "detect_boundaries": m_detect,
            "classify_relevance": m_classify,
            "summarize_report": m_summarize,
        }


def run_pipeline(tmp_path, mocks, **kwargs):
    """Helper that calls process_pdf with sensible defaults."""
    defaults = dict(
        pdf_source=b"fake pdf bytes",
        summary_prompt="Summarize this report.",
        api_key="sk-test-key",
        output_path=str(tmp_path / "results.json"),
        progress_file=str(tmp_path / "progress.json"),
    )
    defaults.update(kwargs)
    return process_pdf(**defaults)


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------

class TestProcessPdfOutput:
    def test_returns_list_of_report_summaries(self, tmp_path, mocks):
        result = run_pipeline(tmp_path, mocks)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ReportSummary)

    def test_writes_output_json_file(self, tmp_path, mocks):
        output = str(tmp_path / "results.json")
        run_pipeline(tmp_path, mocks, output_path=output)
        assert os.path.exists(output)

    def test_output_json_is_parseable_list(self, tmp_path, mocks):
        output = str(tmp_path / "results.json")
        run_pipeline(tmp_path, mocks, output_path=output)
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_output_json_contains_summary_fields(self, tmp_path, mocks):
        output = str(tmp_path / "results.json")
        run_pipeline(tmp_path, mocks, output_path=output)
        with open(output) as f:
            data = json.load(f)
        record = data[0]
        assert "title" in record
        assert "summary" in record
        assert "start_page" in record
        assert "end_page" in record
        assert "report_index" in record


# ---------------------------------------------------------------------------
# Progress file (checkpointing)
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_deletes_progress_file_on_success(self, tmp_path, mocks):
        progress = str(tmp_path / "progress.json")
        run_pipeline(tmp_path, mocks, progress_file=progress)
        assert not os.path.exists(progress)

    def test_resumes_skipping_completed_batches(self, tmp_path, mocks):
        """
        When a progress file exists with last_completed_page >= batch_end,
        the pipeline skips that batch — OCR and boundary detection are not called.
        """
        progress = str(tmp_path / "progress.json")
        state = {
            "last_completed_page": 5,   # batch (1-5) is already done
            "summaries": [
                {
                    "report_index": 1,
                    "title": "Pre-existing Report",
                    "start_page": 1,
                    "end_page": 5,
                    "summary": "Already summarized.",
                }
            ],
            "pending_report": None,
        }
        with open(progress, "w") as f:
            json.dump(state, f)

        result = run_pipeline(tmp_path, mocks, progress_file=progress)

        # OCR must not be called since the batch is skipped
        mocks["ocr_batch"].assert_not_called()
        # The pre-existing summary is included in the result
        assert any(s.title == "Pre-existing Report" for s in result)


# ---------------------------------------------------------------------------
# Filtering behaviour
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_skips_report_with_insufficient_text(self, tmp_path, mocks):
        mocks["detect_boundaries"].return_value = [
            ReportBoundary(1, 5, "Report", "too short")   # < 50 chars
        ]
        result = run_pipeline(tmp_path, mocks)
        assert len(result) == 0
        mocks["classify_relevance"].assert_not_called()

    def test_skips_irrelevant_report(self, tmp_path, mocks):
        mocks["classify_relevance"].return_value = False
        result = run_pipeline(tmp_path, mocks)
        assert len(result) == 0
        mocks["summarize_report"].assert_not_called()

    def test_includes_relevant_report(self, tmp_path, mocks):
        mocks["classify_relevance"].return_value = True
        result = run_pipeline(tmp_path, mocks)
        assert len(result) == 1

    def test_deduplicates_overlap_boundary(self, tmp_path, mocks):
        """
        Reports whose start_page was already finalized must not be re-summarized
        when the same page appears in the overlap zone of the next batch.
        """
        batch1 = [(1, 5, [MagicMock()] * 5)]
        batch2 = [(4, 8, [MagicMock()] * 5)]   # overlaps pages 4-5
        mocks["iter_page_batches"].return_value = iter(batch1 + batch2)
        mocks["get_page_count"].return_value = 8

        # Both batches detect a report starting at page 1 (via overlap)
        mocks["detect_boundaries"].return_value = [
            ReportBoundary(1, 5, "Report", "report content text " * 5)
        ]

        result = run_pipeline(tmp_path, mocks)
        # Despite two batches detecting it, it is summarized only once
        assert mocks["summarize_report"].call_count == 1


# ---------------------------------------------------------------------------
# Prompt and claimant handling
# ---------------------------------------------------------------------------

class TestPromptHandling:
    def test_claimant_name_prepended_to_prompt(self, tmp_path, mocks):
        run_pipeline(tmp_path, mocks, claimant_name="Jane Smith")
        call_kwargs = mocks["summarize_report"].call_args[1]
        assert "Jane Smith" in call_kwargs["summary_prompt"]

    def test_no_claimant_prefix_when_empty(self, tmp_path, mocks):
        run_pipeline(tmp_path, mocks, claimant_name="")
        call_kwargs = mocks["summarize_report"].call_args[1]
        assert "Claimant:" not in call_kwargs["summary_prompt"]

    def test_report_indices_sequential(self, tmp_path, mocks):
        """report_index must be 1-based and sequential across all summaries."""
        # Return two boundaries so we get two summaries
        mocks["detect_boundaries"].return_value = [
            ReportBoundary(1, 2, "Report A", "report A text content " * 5),
            ReportBoundary(3, 5, "Report B", "report B text content " * 5),
        ]
        result = run_pipeline(tmp_path, mocks)
        assert [s.report_index for s in result] == [1, 2]


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_is_invoked(self, tmp_path, mocks):
        calls = []
        run_pipeline(tmp_path, mocks, on_progress=lambda f, m: calls.append((f, m)))
        assert len(calls) > 0

    def test_final_callback_fraction_is_one(self, tmp_path, mocks):
        fractions = []
        run_pipeline(tmp_path, mocks, on_progress=lambda f, m: fractions.append(f))
        assert fractions[-1] == 1.0

    def test_fractions_are_between_zero_and_one(self, tmp_path, mocks):
        fractions = []
        run_pipeline(tmp_path, mocks, on_progress=lambda f, m: fractions.append(f))
        assert all(0.0 <= f <= 1.0 for f in fractions)
