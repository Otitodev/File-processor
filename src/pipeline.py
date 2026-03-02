"""
Main pipeline orchestrator.

Processes a large scanned PDF in chunks:
  1. Extract a batch of pages as images.
  2. OCR the images to text.
  3. Detect report boundaries within the batch.
  4. Accumulate partial reports across batch boundaries.
  5. Filter out documents that are not relevant to the medicolegal claim file.
  6. Summarize each completed, relevant report.
  7. Save progress to a JSON file after every batch so the run can be resumed.
"""

import json
import os
from dataclasses import asdict
from typing import Callable, List, Optional, Union

import anthropic

from .extractor import get_page_count, iter_page_batches
from .ocr_engine import OCRBackend, ocr_batch
from .report_analyzer import (
    ReportBoundary,
    ReportSummary,
    classify_relevance,
    detect_boundaries,
    summarize_report,
)


def _load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"last_completed_page": 0, "summaries": [], "pending_report": None}


def _save_progress(progress_file: str, state: dict) -> None:
    with open(progress_file, "w") as f:
        json.dump(state, f, indent=2)


def process_pdf(
    pdf_source: Union[str, bytes],
    summary_prompt: str,
    api_key: str,
    output_path: str = "results.json",
    progress_file: str = ".pipeline_progress.json",
    batch_size: int = 30,
    overlap: int = 2,
    dpi: int = 200,
    ocr_backend: OCRBackend = "tesseract",
    analysis_model: str = "claude-sonnet-4-6",
    claimant_name: str = "",
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> List[ReportSummary]:
    """
    Full pipeline: PDF → list of summarized reports.

    Args:
        pdf_source:      Path to the input PDF, or raw PDF bytes (e.g. downloaded from R2).
        summary_prompt:  Custom prompt for how to summarize each report.
        api_key:         Anthropic API key.
        output_path:     Where to write the final JSON results.
        progress_file:   Checkpoint file for resuming interrupted runs.
        batch_size:      Pages processed per batch. Lower = less RAM, more API calls.
        overlap:         Pages repeated between batches for boundary continuity.
        dpi:             Page rendering resolution for OCR.
        ocr_backend:     "tesseract" (local) or "claude" (vision API, more accurate).
        analysis_model:  Claude model for boundary detection and summarization.
        claimant_name:   Claimant's full name — injected into prompts for context and
                         used by the relevance filter.
        on_progress:     Optional callback(fraction, message) called at each pipeline
                         stage. fraction is 0.0–1.0. Falls back to print() when None.

    Returns:
        List of ReportSummary objects for relevant documents only.
    """

    def _log(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)
        else:
            print(message)

    client = anthropic.Anthropic(api_key=api_key)
    total_pages = get_page_count(pdf_source)
    state = _load_progress(progress_file)

    summaries: List[ReportSummary] = [
        ReportSummary(**s) for s in state.get("summaries", [])
    ]
    # Restore any report that was still open at the end of the last batch
    pending_boundary: Optional[ReportBoundary] = (
        ReportBoundary(**state["pending_report"]) if state.get("pending_report") else None
    )
    last_completed_page: int = state.get("last_completed_page", 0)
    report_counter = len(summaries)

    # Prepend the claimant name to the summary prompt so Claude has full context
    effective_prompt = summary_prompt
    if claimant_name:
        effective_prompt = f"Claimant: {claimant_name}\n\n{summary_prompt}"

    _log(0.0, f"PDF has {total_pages} pages. Resuming from page {last_completed_page + 1}.")

    # Track finalized start pages to prevent duplicates from the overlap zone.
    # Seed from restored summaries so a resumed run doesn't re-finalize them.
    finalized_start_pages: set = {s.start_page for s in summaries}

    def _finalize(boundary: ReportBoundary, fraction: float) -> None:
        """Classify relevance and, if relevant, summarize a complete report."""
        nonlocal report_counter
        # Skip reports with no usable OCR text — avoids "I don't see any report text" responses
        if len(boundary.text.strip()) < 50:
            _log(fraction, f"Skipped (no text): {boundary.title}")
            return
        # Skip if this start page was already finalized (overlap duplicate)
        if boundary.start_page in finalized_start_pages:
            _log(fraction, f"Skipped (duplicate at page {boundary.start_page}): {boundary.title}")
            return
        finalized_start_pages.add(boundary.start_page)
        if classify_relevance(boundary, claimant_name, client, analysis_model):
            summary = summarize_report(
                report=boundary,
                summary_prompt=effective_prompt,
                client=client,
                model=analysis_model,
            )
            report_counter += 1
            summary.report_index = report_counter
            summaries.append(summary)
            _log(fraction, f"Summarized: {summary.title}")
        else:
            _log(fraction, f"Skipped (not relevant): {boundary.title}")

    batch_iter = iter_page_batches(pdf_source, batch_size=batch_size, dpi=dpi, overlap=overlap)

    for batch_start, batch_end, images in batch_iter:
        # Skip batches we've already processed (resume support)
        if batch_end <= last_completed_page:
            continue

        fraction = batch_end / total_pages

        # ------------------------------------------------------------------
        # Step 1: OCR
        # ------------------------------------------------------------------
        _log(fraction, f"OCR: pages {batch_start}–{batch_end} of {total_pages}")
        page_texts = ocr_batch(images, backend=ocr_backend, client=client)

        # ------------------------------------------------------------------
        # Step 2: Detect report boundaries in this batch
        # ------------------------------------------------------------------
        _log(fraction, f"Detecting boundaries in pages {batch_start}–{batch_end}…")
        boundaries = detect_boundaries(
            page_texts=page_texts,
            batch_start_page=batch_start,
            client=client,
            model=analysis_model,
        )

        # ------------------------------------------------------------------
        # Step 3: Merge with any pending report from the previous batch
        # ------------------------------------------------------------------
        if boundaries:
            first = boundaries[0]

            if pending_boundary is not None:
                if first.start_page <= last_completed_page + overlap:
                    # The first detection is a continuation of the pending report
                    pending_boundary.text += "\n\n" + first.text
                    if first.end_page is not None:
                        # The pending report is now complete — finalize it
                        pending_boundary.end_page = first.end_page
                        _finalize(pending_boundary, fraction)
                        pending_boundary = None
                    # Process remaining boundaries normally
                    remaining = boundaries[1:]
                else:
                    # The pending report ended before this batch — close it
                    pending_boundary.end_page = batch_start - 1
                    _finalize(pending_boundary, fraction)
                    pending_boundary = None
                    remaining = boundaries
            else:
                remaining = boundaries

            # Finalize all fully-bounded reports; hold the last open one
            for boundary in remaining:
                if boundary.end_page is not None:
                    _finalize(boundary, fraction)
                else:
                    # This report continues into the next batch
                    pending_boundary = boundary

        # ------------------------------------------------------------------
        # Step 4: Save progress checkpoint
        # ------------------------------------------------------------------
        last_completed_page = batch_end
        state = {
            "last_completed_page": last_completed_page,
            "summaries": [asdict(s) for s in summaries],
            "pending_report": asdict(pending_boundary) if pending_boundary else None,
        }
        _save_progress(progress_file, state)

    # ------------------------------------------------------------------
    # Step 5: Close any report still open at the very end of the PDF
    # ------------------------------------------------------------------
    if pending_boundary is not None:
        pending_boundary.end_page = total_pages
        _finalize(pending_boundary, 1.0)

    # ------------------------------------------------------------------
    # Step 6: Write final output
    # ------------------------------------------------------------------
    output = [asdict(s) for s in summaries]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint file on successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)

    _log(1.0, f"Done. Found {len(summaries)} relevant report(s). Results saved to: {output_path}")
    return summaries
