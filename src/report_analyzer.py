"""
Report analyzer — uses Claude to:
  1. Detect report boundaries within a chunk of OCR'd pages.
  2. Summarize each identified report using the caller-supplied prompt.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10  # seconds; doubles each attempt (10, 20, 40, 80, 160)


def _create_with_retry(client: anthropic.Anthropic, **kwargs):
    """
    Call client.messages.create with exponential backoff on rate-limit and
    overload errors. Re-raises immediately on any other error type.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < _MAX_RETRIES:  # API overloaded
                time.sleep(delay)
                delay *= 2
            else:
                raise

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

    response = _create_with_retry(
        client,
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
# Relevance classification
# ---------------------------------------------------------------------------

_RELEVANCE_SYSTEM = """\
You are reviewing detected medical documents for an Ontario auto insurance personal injury
claim file. Determine whether this document is RELEVANT for inclusion in a medicolegal
claim file summary.

RELEVANT — include these document types:
- Insurer's examinations / independent medical examinations (IME)
- OCF-18 Treatment and Assessment Plans
- Occupational therapy assessments and reports
- Physiotherapy assessments and reports
- Psychology or psychiatry assessments and reports
- Specialist medical opinions (orthopaedic, neurology, physiatry, pain medicine, etc.)
- Disability Certificates (OCF-3)
- Functional capacity evaluations
- Neuropsychological assessments
- Catastrophic impairment assessments

NOT RELEVANT — exclude these document types:
- Hospital admission or discharge summaries and nursing notes
- Pharmacy records or medication lists
- Basic lab results or diagnostic imaging readings (X-ray, MRI, CT) without specialist
  interpretation
- General family physician clinical or office notes
- Administrative records and correspondence

Return ONLY a JSON object with no markdown fences:
{"relevant": true, "reason": "brief explanation"}
"""


_RELEVANCE_MODEL = "claude-haiku-4-5-20251001"  # yes/no decision — no need for Sonnet


def classify_relevance(
    report: ReportBoundary,
    claimant_name: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",  # kept for signature compat; ignored internally
) -> bool:
    """
    Return True if this report should be included in the medicolegal summary.

    Always uses Haiku — the task is a simple yes/no classification that does not
    need a larger model. Only the first 3 000 characters of the report text are
    sent; the document type is usually apparent from the header alone.
    Returns True (include) if the Claude response cannot be parsed.
    """
    snippet = report.text[:3_000]
    user_msg = (
        f"Claimant: {claimant_name or 'Unknown'}\n"
        f"Detected document title: {report.title}\n\n"
        f"--- DOCUMENT TEXT (first portion) ---\n{snippet}"
    )
    response = _create_with_retry(
        client,
        model=_RELEVANCE_MODEL,
        max_tokens=256,
        system=_RELEVANCE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        return bool(result.get("relevant", True))
    except json.JSONDecodeError:
        return True  # include by default if parsing fails


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = """\
You are a medical-legal document summarizer specializing in insurance and personal injury claims (Ontario auto insurance context).

You will receive OCR-extracted text from a single medical document or report. Produce a concise prose summary following these rules exactly:

FORMAT RULES
- Write in clear, professional prose paragraphs — no bullet points, no headings.
- Only output the summary itself; no preamble, no commentary.

CONTENT RULES
1. Open with the author's full name and credential/role (e.g., "Dr. Mohamed Khaled, Physician," or
   "Laura Nelson, Occupational Therapist (College Registration Number G1911702),").
   - If the document has no identifiable author (e.g., an OCF-3 form), omit the author line.
2. State what the author did: "completed", "authored", "prepared", etc.
3. Name the document type exactly as it appears in the document (e.g.,
   INSURER'S EXAMINATION – MEDICAL PHYSICIAN ASSESSMENT, OCF-18 Treatment and Assessment Plan,
   Occupational Therapy Initial Report, Disability Certificate (OCF-3)).
   - For OCF-18 plans include the effective date in parentheses, e.g. "(Effective date 2016-10-01)".
4. Include the document date (e.g., "dated December 30, 2025").
5. Summarize the key content:
   - For independent medical examinations (IME) / insurer's examinations: describe the clinical
     findings, diagnosed injuries, accident causation opinions, functional limitations, aids used,
     and any pre-existing condition findings. If the examiner also opines on a disputed OCF-18,
     address that in a separate paragraph starting "With respect to the disputed OCF-18 dated
     [date], in the amount of $[amount], [Author last name] opined that…", and list the services.
   - For OCF-18 Treatment and Assessment Plans: state the total dollar amount, the proposed
     service categories, and whether the plan was approved and for what amount.
   - For clinical/therapy reports (OT, PT, psychology, etc.): describe the claimant's functional
     status, impairments, symptoms, and any recommended interventions or equipment.
   - For Disability Certificates (OCF-3): list the accident-related injuries identified, and
     describe the functional limitations documented (complete inability to carry on normal life,
     substantial inability to perform housekeeping, etc.).
6. Refer to the author by last name (with appropriate title) after the first mention.
7. Do not invent or extrapolate information that is not in the document text.
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

    response = _create_with_retry(
        client,
        model=model,
        max_tokens=2048,
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
