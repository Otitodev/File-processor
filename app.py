"""
Streamlit web UI for the Medical Document Analyzer.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from src.pipeline import process_pdf
from src.docx_writer import generate_word_document

# ---------------------------------------------------------------------------
# Default summarization prompt
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = """\
Summarize this medical document for use in an Ontario auto insurance personal injury \
claim file. Follow the format rules exactly as instructed."""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Document Analyzer",
    page_icon="⚕",
    layout="centered",
)

st.title("Medical Document Analyzer")
st.caption(
    "Processes scanned medical PDFs, filters to relevant reports, "
    "and generates a formatted Word summary."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.subheader("Claimant & Files")

claimant_name = st.text_input(
    "Claimant name",
    placeholder="e.g. Jane Smith",
    help="Used to identify the claimant in every prompt sent to Claude.",
)

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="You may upload one or more scanned PDF disclosure packages.",
)

st.subheader("Summarization Prompt")

summary_prompt = st.text_area(
    "Prompt sent to Claude for each relevant report",
    value=_DEFAULT_PROMPT,
    height=120,
    help=(
        "Edit this prompt to change how each report is summarized. "
        "The claimant's name is automatically prepended."
    ),
)

st.subheader("Settings")

col1, col2 = st.columns(2)

with col1:
    ocr_backend = st.selectbox(
        "OCR backend",
        options=["tesseract", "claude"],
        help=(
            "tesseract — free, fast, local. "
            "claude — uses Claude Vision (more accurate for poor scans, costs image tokens)."
        ),
    )

with col2:
    analysis_model = st.selectbox(
        "Claude model",
        options=["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        help="Model used for boundary detection, relevance filtering, and summarization.",
    )

api_key = st.text_input(
    "Anthropic API key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    help="Falls back to the ANTHROPIC_API_KEY environment variable if set.",
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ready = bool(claimant_name.strip() and uploaded_files and summary_prompt.strip() and api_key.strip())

if not claimant_name.strip():
    st.info("Enter the claimant's name to continue.")
elif not uploaded_files:
    st.info("Upload at least one PDF file to continue.")
elif not api_key.strip():
    st.warning("An Anthropic API key is required.")

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if st.button("Process documents", disabled=not ready, type="primary"):
    all_summaries = []
    total_files = len(uploaded_files)

    overall_bar = st.progress(0, text="Starting…")
    status_box = st.empty()

    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_label = uploaded_file.name

        # Write the uploaded bytes to a temp file the pipeline can read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Unique progress file per upload so parallel runs don't collide
        progress_file = f".pipeline_progress_{file_idx}.json"

        try:
            file_summaries_before = len(all_summaries)

            def make_progress_callback(f_idx, f_label, f_total):
                """Returns a closure bound to the current file's context."""
                def _on_progress(fraction: float, message: str) -> None:
                    # Overall fraction: completed files + current file progress
                    overall = (f_idx + fraction) / f_total
                    overall_bar.progress(
                        min(overall, 1.0),
                        text=f"File {f_idx + 1}/{f_total} — {f_label}: {message}",
                    )
                    status_box.markdown(f"**{f_label}** — {message}")
                return _on_progress

            file_summaries = process_pdf(
                pdf_path=tmp_path,
                summary_prompt=summary_prompt,
                api_key=api_key,
                output_path=os.devnull,       # UI handles saving; skip the JSON file
                progress_file=progress_file,
                ocr_backend=ocr_backend,
                analysis_model=analysis_model,
                claimant_name=claimant_name.strip(),
                on_progress=make_progress_callback(file_idx, file_label, total_files),
            )

            # Offset indices so they're globally sequential across all files
            offset = len(all_summaries)
            for s in file_summaries:
                s.report_index = offset + s.report_index

            all_summaries.extend(file_summaries)
            new_count = len(all_summaries) - file_summaries_before
            status_box.markdown(
                f"**{file_label}** — done. {new_count} relevant report(s) found."
            )

        finally:
            os.unlink(tmp_path)
            if os.path.exists(progress_file):
                os.remove(progress_file)

    overall_bar.progress(1.0, text="All files processed.")

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------

    if not all_summaries:
        st.warning(
            "No relevant medical reports were found in the uploaded file(s). "
            "All detected documents were filtered out by the relevance check."
        )
    else:
        st.success(f"Found **{len(all_summaries)}** relevant report(s).")

        st.subheader("Summaries")
        for s in all_summaries:
            with st.expander(f"[{s.report_index}] {s.title}  (pages {s.start_page}–{s.end_page})"):
                st.write(s.summary)

        # -------------------------------------------------------------------
        # Word document download
        # -------------------------------------------------------------------
        st.subheader("Download")

        docx_bytes = generate_word_document(claimant_name.strip(), all_summaries)
        safe_name = claimant_name.strip().replace(" ", "_")
        st.download_button(
            label="Download Word document (.docx)",
            data=docx_bytes,
            file_name=f"{safe_name}_medical_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
