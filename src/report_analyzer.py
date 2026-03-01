"""
Report analyzer — uses Claude to:
  1. Detect report boundaries within a chunk of OCR'd pages.
  2. Summarize each identified report using the caller-supplied prompt.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReportBoundary:
    """A report detected within a batch of pages."""
    start_page: int
    end_page: Optional[int]   # None means "continues into the next batch"
    title: str                # Best guess at the report title / type
    text: str                 # Concatenated OCR text for this report


@dataclass
class ReportSummary:
    """Final output for a single medical report."""
    report_index: int
    title: str
    start_page: int
    end_page: int
    summary: str


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

_BOUNDARY_SYSTEM = """\
You are a medical document analyst. You will receive OCR-extracted text from a batch of
consecutive pages of a scanned PDF. The PDF contains multiple distinct medical reports
concatenated together.

Your task: identify where each medical report starts and ends within this batch.

Return a JSON array (no markdown, raw JSON only) with one object per report found:
[
  {
    "title": "<report type or patient name if visible>",
    "start_page": <1-based page number within this batch>,
    "end_page": <1-based page number, or null if the report continues beyond this batch>
  }
]

Rules:
- A new report typically begins with a header (patient name, date, report type, facility name, etc.)
- If the batch starts mid-report, use start_page = 1.
- If a report ends after the last page in the batch, set end_page to null.
- Do NOT output anything except the JSON array.
"""


def detect_boundaries(
    page_texts: List[str],
    batch_start_page: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
) -> List[ReportBoundary]:
    """
    Ask Claude to identify individual report boundaries within a batch of pages.

    Args:
        page_texts:       List of OCR text strings (one per page in this batch).
        batch_start_page: The absolute 1-based page number of the first page in this batch.
        client:           Anthropic client.
        model:            Claude model to use.

    Returns:
        List of ReportBoundary objects with absolute page numbers.
    """
    # Build the prompt: include page numbers so Claude can reference them
    pages_block = ""
    for i, text in enumerate(page_texts):
        abs_page = batch_start_page + i
        pages_block += f"\n\n--- PAGE {abs_page} ---\n{text.strip()}"

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_BOUNDARY_SYSTEM,
        messages=[{"role": "user", "content": pages_block}],
    )

    raw = response.content[0].text.strip()

    # Strip potential markdown fences Claude sometimes adds despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        boundaries = json.loads(raw)
    except json.JSONDecodeError:
        # If parsing fails, treat the entire batch as one unknown report
        boundaries = [{"title": "Unknown Report", "start_page": 1, "end_page": None}]

    results = []
    for b in boundaries:
        # Relative page numbers from Claude → absolute page numbers
        rel_start = b.get("start_page", 1)
        rel_end = b.get("end_page")

        abs_start = batch_start_page + rel_start - 1
        abs_end = (batch_start_page + rel_end - 1) if rel_end is not None else None

        # Collect the text belonging to this report
        text_start_idx = rel_start - 1
        text_end_idx = rel_end if rel_end is not None else len(page_texts)
        report_text = "\n\n".join(page_texts[text_start_idx:text_end_idx])

        results.append(
            ReportBoundary(
                start_page=abs_start,
                end_page=abs_end,
                title=b.get("title", "Unknown Report"),
                text=report_text,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = """\
You are a medical document summarizer. You will receive OCR-extracted text from a single
medical report (possibly spanning multiple pages). Summarize it according to the user's
instructions. Be accurate and concise. Only output the summary, no preamble.
"""


def summarize_report(
    report: ReportBoundary,
    summary_prompt: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    max_text_chars: int = 60_000,
) -> ReportSummary:
    """
    Summarize a single report using the caller-supplied prompt.

    Args:
        report:          The report whose text will be summarized.
        summary_prompt:  The user's custom summarization instruction.
        client:          Anthropic client.
        model:           Claude model to use.
        max_text_chars:  Truncate report text to this length to avoid exceeding context.
                         Increase if your model supports a larger context window.

    Returns:
        A ReportSummary with the generated summary text.
    """
    text = report.text
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + "\n\n[...text truncated due to length...]"

    user_message = f"{summary_prompt}\n\n--- REPORT TEXT ---\n{text}"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SUMMARIZE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    return ReportSummary(
        report_index=0,          # set by the caller
        title=report.title,
        start_page=report.start_page,
        end_page=report.end_page or report.start_page,
        summary=response.content[0].text.strip(),
    )
