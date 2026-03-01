"""
Main pipeline orchestrator.

Processes a large scanned PDF in chunks:
  1. Extract a batch of pages as images.
  2. OCR the images to text.
  3. Detect report boundaries within the batch.
  4. Accumulate partial reports across batch boundaries.
  5. Summarize each completed report.
  6. Save progress to a JSON file after every batch so the run can be resumed.
"""

import json
import os
from dataclasses import asdict
from typing import List, Optional

import anthropic
from tqdm import tqdm

from .extractor import get_page_count, iter_page_batches
from .ocr_engine import OCRBackend, ocr_batch
from .report_analyzer import (
    ReportBoundary,
    ReportSummary,
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
    pdf_path: str,
    summary_prompt: str,
    api_key: str,
    output_path: str = "results.json",
    progress_file: str = ".pipeline_progress.json",
    batch_size: int = 30,
    overlap: int = 2,
    dpi: int = 200,
    ocr_backend: OCRBackend = "tesseract",
    analysis_model: str = "claude-sonnet-4-6",
) -> List[ReportSummary]:
    """
    Full pipeline: PDF → list of summarized reports.

    Args:
        pdf_path:        Path to the input PDF.
        summary_prompt:  Your custom prompt for how to summarize each report.
        api_key:         Anthropic API key.
        output_path:     Where to write the final JSON results.
        progress_file:   Checkpoint file for resuming interrupted runs.
        batch_size:      Pages processed per batch. Lower = less RAM, more API calls.
        overlap:         Pages repeated between batches for boundary continuity.
        dpi:             Page rendering resolution for OCR.
        ocr_backend:     "tesseract" (local) or "claude" (vision API, more accurate).
        analysis_model:  Claude model for boundary detection and summarization.

    Returns:
        List of ReportSummary objects.
    """
    client = anthropic.Anthropic(api_key=api_key)
    total_pages = get_page_count(pdf_path)
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

    print(f"PDF has {total_pages} pages. Resuming from page {last_completed_page + 1}.")

    batch_iter = iter_page_batches(pdf_path, batch_size=batch_size, dpi=dpi, overlap=overlap)

    with tqdm(total=total_pages, initial=last_completed_page, unit="page", desc="Processing") as pbar:
        for batch_start, batch_end, images in batch_iter:
            # Skip batches we've already processed (resume support)
            if batch_end <= last_completed_page:
                continue

            # ------------------------------------------------------------------
            # Step 1: OCR
            # ------------------------------------------------------------------
            page_texts = ocr_batch(images, backend=ocr_backend, client=client)

            # ------------------------------------------------------------------
            # Step 2: Detect report boundaries in this batch
            # ------------------------------------------------------------------
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
                            # The pending report is now complete — summarize it
                            pending_boundary.end_page = first.end_page
                            summary = summarize_report(
                                report=pending_boundary,
                                summary_prompt=summary_prompt,
                                client=client,
                                model=analysis_model,
                            )
                            report_counter += 1
                            summary.report_index = report_counter
                            summaries.append(summary)
                            pending_boundary = None
                        # Process remaining boundaries normally
                        remaining = boundaries[1:]
                    else:
                        # The pending report ended before this batch — close it
                        pending_boundary.end_page = batch_start - 1
                        summary = summarize_report(
                            report=pending_boundary,
                            summary_prompt=summary_prompt,
                            client=client,
                            model=analysis_model,
                        )
                        report_counter += 1
                        summary.report_index = report_counter
                        summaries.append(summary)
                        pending_boundary = None
                        remaining = boundaries
                else:
                    remaining = boundaries

                # Summarize all fully-bounded reports; hold the last open one
                for boundary in remaining:
                    if boundary.end_page is not None:
                        summary = summarize_report(
                            report=boundary,
                            summary_prompt=summary_prompt,
                            client=client,
                            model=analysis_model,
                        )
                        report_counter += 1
                        summary.report_index = report_counter
                        summaries.append(summary)
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

            pbar.update(batch_end - (batch_start - (overlap if batch_start > 1 else 0)))

    # ------------------------------------------------------------------
    # Step 5: Close any report still open at the very end of the PDF
    # ------------------------------------------------------------------
    if pending_boundary is not None:
        pending_boundary.end_page = total_pages
        summary = summarize_report(
            report=pending_boundary,
            summary_prompt=summary_prompt,
            client=client,
            model=analysis_model,
        )
        report_counter += 1
        summary.report_index = report_counter
        summaries.append(summary)

    # ------------------------------------------------------------------
    # Step 6: Write final output
    # ------------------------------------------------------------------
    output = [asdict(s) for s in summaries]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint file on successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"\nDone. Found {len(summaries)} report(s). Results saved to: {output_path}")
    return summaries
