"""
Streamlit web UI for the Medical Document Analyzer.

Run with:
    streamlit run app.py

R2 storage is optional. When R2 credentials are provided (via UI or
.streamlit/secrets.toml), uploaded PDFs are stored in Cloudflare R2 and
processed from there — no temp files on disk. Without R2 credentials, the
uploaded file bytes are passed directly to the pipeline (no temp files either).
"""

import io
import json
import os

import anthropic
import streamlit as st

from src.pipeline import process_pdf
from src.report_analyzer import ReportSummary
from src.docx_writer import generate_word_document
from src.db import init_db, save_run, list_runs, get_run_summaries, delete_run

init_db()


def _recover_partial_summaries(progress_file: str, offset: int) -> list:
    """Read any summaries already saved in a checkpoint file."""
    if not os.path.exists(progress_file):
        return []
    try:
        with open(progress_file) as f:
            state = json.load(f)
        saved = state.get("summaries", [])
        if not saved:
            return []
        recovered = []
        for s in saved:
            rs = ReportSummary(**s)
            rs.report_index = offset + rs.report_index
            recovered.append(rs)
        return recovered
    except Exception:
        return []

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
# Session state defaults
# ---------------------------------------------------------------------------

if "r2_files" not in st.session_state:
    # List of dicts: {object_key, filename, size_bytes}
    st.session_state["r2_files"] = []

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
# R2 configuration (optional)
# ---------------------------------------------------------------------------

with st.expander("Cloudflare R2 storage (optional — recommended for large files)"):
    st.caption(
        "When configured, PDFs are uploaded to R2 and processed from there — "
        "no temp files on the server. Files are kept in R2 after processing "
        "so you can re-run without re-uploading."
    )

    try:
        r2_secrets = st.secrets.get("r2", {})
    except Exception:
        r2_secrets = {}

    r2_account_id = st.text_input(
        "R2 Account ID",
        value=r2_secrets.get("account_id", "") or os.environ.get("R2_ACCOUNT_ID", ""),
        help="Found in the Cloudflare dashboard → R2 → Manage R2 API tokens.",
    )
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        r2_access_key = st.text_input(
            "R2 Access Key ID",
            value=r2_secrets.get("access_key_id", "") or os.environ.get("R2_ACCESS_KEY_ID", ""),
        )
    with r2_col2:
        r2_secret_key = st.text_input(
            "R2 Secret Access Key",
            type="password",
            value=r2_secrets.get("secret_access_key", "") or os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        )
    r2_bucket = st.text_input(
        "R2 Bucket name",
        value=r2_secrets.get("bucket_name", "") or os.environ.get("R2_BUCKET_NAME", ""),
    )
    delete_after_processing = st.checkbox(
        "Delete from R2 after processing",
        value=False,
        help="If unchecked, files stay in R2 — useful for re-running without re-uploading.",
    )

r2_configured = bool(r2_account_id and r2_access_key and r2_secret_key and r2_bucket)


def _get_r2_client():
    from src.r2_storage import R2Config, get_client
    cfg = R2Config(
        account_id=r2_account_id,
        access_key_id=r2_access_key,
        secret_access_key=r2_secret_key,
        bucket_name=r2_bucket,
    )
    return get_client(cfg)


# ---------------------------------------------------------------------------
# Upload files to R2 (when R2 is configured)
# ---------------------------------------------------------------------------

if r2_configured and uploaded_files:
    from src.r2_storage import make_object_key, upload_fileobj

    already_staged = {f["filename"] for f in st.session_state["r2_files"]}
    new_files = [f for f in uploaded_files if f.name not in already_staged]

    if new_files:
        client = _get_r2_client()
        for uf in new_files:
            with st.spinner(f"Uploading {uf.name} to R2…"):
                try:
                    key = make_object_key(claimant_name, uf.name)
                    upload_fileobj(client, r2_bucket, io.BytesIO(uf.read()), key)
                    st.session_state["r2_files"].append(
                        {"object_key": key, "filename": uf.name, "size_bytes": uf.size}
                    )
                except Exception as e:
                    st.error(f"Failed to upload **{uf.name}** to R2: {e}")
        uploaded_count = sum(
            1 for f in st.session_state["r2_files"] if f["filename"] not in already_staged
        )
        if uploaded_count:
            st.success(f"Uploaded {uploaded_count} file(s) to R2.")

# ---------------------------------------------------------------------------
# Manage R2 files
# ---------------------------------------------------------------------------

if r2_configured and st.session_state["r2_files"]:
    st.subheader("Files in R2")
    to_remove = []
    for i, r2f in enumerate(st.session_state["r2_files"]):
        col_name, col_size, col_btn = st.columns([4, 2, 1])
        col_name.write(r2f["filename"])
        col_size.write(f"{r2f['size_bytes'] / 1_048_576:.1f} MB")
        if col_btn.button("🗑", key=f"del_r2_{i}", help="Delete from R2"):
            from src.r2_storage import delete_object
            try:
                client = _get_r2_client()
                delete_object(client, r2_bucket, r2f["object_key"])
                to_remove.append(i)
            except Exception as e:
                st.error(f"Could not delete **{r2f['filename']}** from R2: {e}")
    for i in reversed(to_remove):
        st.session_state["r2_files"].pop(i)
    if to_remove:
        st.rerun()

# ---------------------------------------------------------------------------
# Determine files to process
# ---------------------------------------------------------------------------

# When R2 is configured, process what's in R2. Otherwise use the uploader directly.
files_ready_for_processing = (
    len(st.session_state["r2_files"]) > 0
    if r2_configured
    else bool(uploaded_files)
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ready = bool(
    claimant_name.strip()
    and files_ready_for_processing
    and summary_prompt.strip()
    and api_key.strip()
)

if not claimant_name.strip():
    st.info("Enter the claimant's name to continue.")
elif not files_ready_for_processing:
    st.info("Upload at least one PDF file to continue.")
elif not api_key.strip():
    st.warning("An Anthropic API key is required.")

# ---------------------------------------------------------------------------
# Past runs history
# ---------------------------------------------------------------------------

_past_runs = list_runs()
if _past_runs:
    with st.expander(f"Past runs ({len(_past_runs)})"):
        for run in _past_runs:
            col_claimant, col_file, col_date, col_count, col_dl, col_del = st.columns(
                [2, 2, 2, 1, 1, 1]
            )
            col_claimant.write(run["claimant"])
            col_file.write(run["filename"])
            # Show only the date portion of the ISO timestamp
            col_date.write(run["created_at"][:10])
            col_count.write(str(run["report_count"]))

            dl_label = f"dl_{run['id']}"
            del_label = f"del_{run['id']}"

            # Download button — regenerates Word doc from DB on the fly
            try:
                run_summaries = get_run_summaries(run["id"])
                docx_bytes = generate_word_document(run["claimant"], run_summaries)
                safe = run["claimant"].replace(" ", "_")
                col_dl.download_button(
                    label="⬇",
                    data=docx_bytes,
                    file_name=f"{safe}_medical_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=dl_label,
                    help="Download Word document",
                )
            except Exception as e:
                col_dl.write("—")

            # Delete button
            if col_del.button("🗑", key=del_label, help="Delete this run"):
                try:
                    delete_run(run["id"])
                except Exception as e:
                    st.error(f"Could not delete run: {e}")
                st.rerun()

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if st.button("Process documents", disabled=not ready, type="primary"):
    all_summaries = []

    if r2_configured:
        from src.r2_storage import download_as_bytes, delete_object as r2_delete
        work_items = [
            (r2f["filename"], r2f["object_key"], r2f)
            for r2f in st.session_state["r2_files"]
        ]
    else:
        work_items = [
            (uf.name, uf, None)
            for uf in uploaded_files
        ]

    total_files = len(work_items)
    overall_bar = st.progress(0, text="Starting…")
    status_box = st.empty()

    for file_idx, (file_label, source_ref, r2f_meta) in enumerate(work_items):
        progress_file = f".pipeline_progress_{file_idx}.json"

        try:
            file_summaries_before = len(all_summaries)

            def make_progress_callback(f_idx, f_label, f_total):
                def _on_progress(fraction: float, message: str) -> None:
                    overall = (f_idx + fraction) / f_total
                    overall_bar.progress(
                        min(overall, 1.0),
                        text=f"File {f_idx + 1}/{f_total} — {f_label}: {message}",
                    )
                    status_box.markdown(f"**{f_label}** — {message}")
                return _on_progress

            if r2_configured:
                # Download from R2 into memory — no temp file on disk
                try:
                    client_r2 = _get_r2_client()
                    with st.spinner(f"Downloading {file_label} from R2…"):
                        pdf_source = download_as_bytes(client_r2, r2_bucket, source_ref)
                except Exception as e:
                    st.error(f"Failed to download **{file_label}** from R2: {e}")
                    continue
            else:
                # Read uploaded bytes directly — no temp file needed
                pdf_source = source_ref.read()

            file_summaries = process_pdf(
                pdf_source=pdf_source,
                summary_prompt=summary_prompt,
                api_key=api_key,
                output_path=os.devnull,
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

            # Persist to SQLite so results survive page refreshes
            if file_summaries:
                try:
                    save_run(claimant_name.strip(), file_label, file_summaries)
                except Exception as e:
                    st.warning(f"Could not save results to database: {e}")

            # Optionally delete from R2 after successful processing
            if r2_configured and delete_after_processing and r2f_meta is not None:
                try:
                    r2_delete(client_r2, r2_bucket, r2f_meta["object_key"])
                    st.session_state["r2_files"] = [
                        f for f in st.session_state["r2_files"]
                        if f["object_key"] != r2f_meta["object_key"]
                    ]
                except Exception as e:
                    st.warning(
                        f"Could not delete **{file_label}** from R2 after processing: {e}"
                    )

        except anthropic.AuthenticationError:
            st.error(
                "Invalid Anthropic API key — check the key in the Settings section and try again."
            )
            break  # all subsequent files will fail with the same key

        except anthropic.RateLimitError:
            partial = _recover_partial_summaries(progress_file, len(all_summaries))
            if partial:
                all_summaries.extend(partial)
                st.error(
                    f"Anthropic rate limit reached — recovered **{len(partial)}** report(s) "
                    f"processed so far (shown below). Wait a moment, then click "
                    f"**Process documents** again to resume from where it stopped."
                )
            else:
                st.error(
                    "Anthropic rate limit reached. Wait a moment, then click "
                    "**Process documents** again to resume."
                )
            break

        except anthropic.APIStatusError as e:
            partial = _recover_partial_summaries(progress_file, len(all_summaries))
            if partial:
                all_summaries.extend(partial)
            st.error(
                f"Anthropic API error on **{file_label}**: {e.message}"
                + (f" — recovered {len(partial)} report(s) so far." if partial else "")
            )

        except Exception as e:
            partial = _recover_partial_summaries(progress_file, len(all_summaries))
            if partial:
                all_summaries.extend(partial)
            st.error(
                f"Failed to process **{file_label}**: {e}"
                + (f" — recovered {len(partial)} report(s) so far." if partial else "")
            )

        # Note: no finally cleanup here. pipeline.py deletes the progress file on
        # success. On error (rate limit, auth, etc.) we deliberately keep it so
        # that clicking "Process documents" again resumes from the last checkpoint.

    overall_bar.progress(1.0, text="Done.")

    # -----------------------------------------------------------------------
    # Results — shown even if some files failed
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
