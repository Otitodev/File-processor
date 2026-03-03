"""
Streamlit web UI for the Medical Document Analyzer.

Run with:
    streamlit run app.py

All credentials are read from environment variables — never shown in the UI:
    ANTHROPIC_API_KEY     — required
    R2_ENABLED            — set to "true" to enable Cloudflare R2 storage
    R2_ACCOUNT_ID         — Cloudflare account ID
    R2_ACCESS_KEY_ID      — R2 API token access key
    R2_SECRET_ACCESS_KEY  — R2 API token secret key
    R2_BUCKET_NAME        — R2 bucket name

When R2_ENABLED=true, large PDFs are uploaded directly from the browser to R2
via a presigned POST URL, bypassing the Railway reverse proxy entirely.
Without R2, uploaded file bytes are passed directly to the pipeline in memory.
"""

import io
import json
import os
import uuid as _uuid

import anthropic
import streamlit as st

from src.pipeline import process_pdf
from src.report_analyzer import ReportSummary
from src.docx_writer import generate_word_document
from src.db import (
    init_db, save_run, list_runs, get_run_summaries, delete_run,
    list_sessions, get_session_summaries, delete_session,
    save_prompt, list_prompts, delete_prompt,
)

init_db()

# ---------------------------------------------------------------------------
# Credentials — read from environment, never shown in UI
# ---------------------------------------------------------------------------

_api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
_r2_enabled = os.environ.get("R2_ENABLED", "").lower() in ("1", "true", "yes")
_r2_account = os.environ.get("R2_ACCOUNT_ID", "")
_r2_access  = os.environ.get("R2_ACCESS_KEY_ID", "")
_r2_secret  = os.environ.get("R2_SECRET_ACCESS_KEY", "")
_r2_bucket  = os.environ.get("R2_BUCKET_NAME", "")

r2_configured = _r2_enabled and bool(_r2_account and _r2_access and _r2_secret and _r2_bucket)


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


def _get_r2_client():
    from src.r2_storage import R2Config, get_client
    return get_client(R2Config(
        account_id=_r2_account,
        access_key_id=_r2_access,
        secret_access_key=_r2_secret,
        bucket_name=_r2_bucket,
    ))


def _render_r2_upload_widget(presigned: dict, filename: str) -> None:
    """
    Render an HTML/JS upload widget that POSTs a file directly to R2
    using a presigned POST URL. No data passes through Railway's proxy.

    Args:
        presigned: dict with "url" and "fields" from generate_presigned_post().
        filename:  Expected filename, shown as a hint to the user.
    """
    import json as _json

    fields_json = _json.dumps(presigned["fields"])
    post_url    = _json.dumps(presigned["url"])
    hint        = _json.dumps(filename)

    widget_html = f"""
<style>
  #r2box {{
    font-family: sans-serif;
    padding: 12px 14px;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #fafafa;
  }}
  #r2box label {{ display: block; font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #333; }}
  #r2-file   {{ display: block; margin-bottom: 10px; }}
  #r2-btn    {{ background: #1f6aa5; color: #fff; border: none; border-radius: 4px;
                padding: 7px 18px; font-size: 14px; cursor: pointer; }}
  #r2-btn:disabled {{ background: #888; cursor: not-allowed; }}
  #r2-prog-wrap {{ display: none; margin-top: 10px; }}
  #r2-bar-bg  {{ background: #e0e0e0; border-radius: 5px; height: 22px;
                 position: relative; overflow: hidden; }}
  #r2-bar-fill {{ background: #1f6aa5; height: 100%; width: 0%;
                  border-radius: 5px; transition: width 0.15s ease; }}
  #r2-pct     {{ position: absolute; top: 50%; left: 50%;
                 transform: translate(-50%, -50%);
                 font-size: 12px; font-weight: 700; color: #fff;
                 text-shadow: 0 0 3px rgba(0,0,0,0.4); pointer-events: none; }}
  #r2-status {{ margin-top: 6px; font-size: 13px; min-height: 18px; }}
</style>
<div id="r2box">
  <label>Upload PDF directly to R2 (large files supported)</label>
  <input type="file" id="r2-file" accept=".pdf" />
  <button id="r2-btn" onclick="doUpload()">Upload to R2</button>
  <div id="r2-prog-wrap">
    <div id="r2-bar-bg">
      <div id="r2-bar-fill"></div>
      <span id="r2-pct">0%</span>
    </div>
  </div>
  <div id="r2-status">Select the file <b>{filename}</b> then click Upload.</div>
</div>
<script>
const POST_URL = {post_url};
const FIELDS   = {fields_json};
const HINT     = {hint};

function doUpload() {{
  const inp      = document.getElementById('r2-file');
  const stat     = document.getElementById('r2-status');
  const progWrap = document.getElementById('r2-prog-wrap');
  const barFill  = document.getElementById('r2-bar-fill');
  const pctLabel = document.getElementById('r2-pct');
  const btn      = document.getElementById('r2-btn');

  if (!inp.files || !inp.files.length) {{
    stat.style.color = '#c00';
    stat.textContent = 'Please select a PDF file first.';
    return;
  }}

  const file = inp.files[0];
  const form = new FormData();
  // Presigned fields must come before the file (S3/R2 spec)
  for (const [k, v] of Object.entries(FIELDS)) form.append(k, v);
  form.append('file', file);

  btn.disabled = true;
  progWrap.style.display = 'block';
  barFill.style.width = '0%';
  pctLabel.textContent = '0%';
  stat.style.color = '#333';
  stat.textContent = 'Uploading\u2026';

  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener('progress', e => {{
    if (e.lengthComputable) {{
      const pct = Math.round(e.loaded / e.total * 100);
      barFill.style.width = pct + '%';
      pctLabel.textContent = pct + '%';
      stat.textContent = (e.loaded / 1048576).toFixed(1) + ' MB / '
        + (e.total / 1048576).toFixed(1) + ' MB';
    }}
  }});

  xhr.addEventListener('load', () => {{
    btn.disabled = false;
    progWrap.style.display = 'none';
    if (xhr.status >= 200 && xhr.status < 300) {{
      stat.style.color = '#1a7a1a';
      stat.textContent = '\u2713 Upload complete! Click \u201cConfirm upload\u201d below.';
    }} else {{
      stat.style.color = '#c00';
      stat.textContent = 'Upload failed (HTTP ' + xhr.status + '). '
        + xhr.responseText.substring(0, 200);
    }}
  }});

  xhr.addEventListener('error', () => {{
    btn.disabled = false;
    progWrap.style.display = 'none';
    stat.style.color = '#c00';
    stat.textContent = 'Network error. Check your connection and that the R2 bucket '
      + 'CORS policy allows this origin.';
  }});

  xhr.open('POST', POST_URL);
  xhr.send(form);
}}
</script>
"""
    st.components.v1.html(widget_html, height=200, scrolling=False)


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

if "pending_upload" not in st.session_state:
    # Holds presigned POST state between Streamlit reruns while user uploads
    st.session_state["pending_upload"] = None

if "summary_prompt" not in st.session_state:
    st.session_state["summary_prompt"] = _DEFAULT_PROMPT

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.subheader("Claimant & Files")

claimant_name = st.text_input(
    "Claimant name",
    placeholder="e.g. Jane Smith",
    help="Used to identify the claimant in every prompt sent to Claude.",
)

# When R2 is enabled, the file uploader is replaced by the presigned POST flow
# below (the native uploader would fail for large files via Railway's proxy).
if not r2_configured:
    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You may upload one or more scanned PDF disclosure packages.",
    )
else:
    uploaded_files = []   # not used in the R2 path

st.subheader("Summarization Prompt")

# --- Load a saved prompt ---
_saved_prompts = list_prompts()
if _saved_prompts:
    with st.expander(f"Saved prompts ({len(_saved_prompts)})"):
        for _sp in _saved_prompts:
            _col_name, _col_load, _col_del = st.columns([5, 1, 1])
            _col_name.write(_sp["name"])
            if _col_load.button("Load", key=f"load_prompt_{_sp['id']}"):
                st.session_state["summary_prompt"] = _sp["text"]
                st.rerun()
            if _col_del.button("🗑", key=f"del_prompt_{_sp['id']}", help="Delete this prompt"):
                delete_prompt(_sp["id"])
                st.rerun()

# --- Edit area (key keeps value in sync with session state) ---
summary_prompt = st.text_area(
    "Prompt sent to Claude for each relevant report",
    key="summary_prompt",
    height=250,
    help=(
        "Edit this prompt to change how each report is summarized. "
        "The claimant's name is automatically prepended."
    ),
)

# --- Save current prompt ---
_col_pname, _col_psave = st.columns([4, 1])
with _col_pname:
    _prompt_save_name = st.text_input(
        "Save prompt as",
        placeholder="e.g. Standard auto insurance",
        label_visibility="collapsed",
    )
with _col_psave:
    if st.button("Save prompt", disabled=not _prompt_save_name.strip()):
        save_prompt(_prompt_save_name.strip(), summary_prompt)
        st.success(f'Saved "{_prompt_save_name.strip()}"')
        st.rerun()

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

# ---------------------------------------------------------------------------
# R2 direct upload (when R2_ENABLED=true)
# ---------------------------------------------------------------------------

if r2_configured:
    from src.r2_storage import make_object_key, generate_presigned_post

    st.subheader("Upload file to R2")

    if not claimant_name.strip():
        st.info("Enter the claimant's name above before uploading files.")
    else:
        pdf_filename = st.text_input(
            "PDF filename",
            placeholder="e.g. Ashok_Kapoor_Med_File.pdf",
            help="Type the filename of the PDF you are about to upload.",
        ).strip()

        if pdf_filename:
            pending = st.session_state["pending_upload"]

            # Generate (or reuse) presigned POST — only regenerate when
            # filename or claimant name changes to avoid flickering on reruns.
            needs_new = (
                pending is None
                or pending["filename"] != pdf_filename
                or pending["claimant_name"] != claimant_name.strip()
            )
            if needs_new:
                client = _get_r2_client()
                object_key = make_object_key(claimant_name.strip(), pdf_filename)
                presigned  = generate_presigned_post(client, _r2_bucket, object_key)
                st.session_state["pending_upload"] = {
                    "filename":      pdf_filename,
                    "object_key":    object_key,
                    "presigned":     presigned,
                    "claimant_name": claimant_name.strip(),
                }
                pending = st.session_state["pending_upload"]

            _render_r2_upload_widget(pending["presigned"], pdf_filename)

            if st.button("✓ Confirm upload", key="r2_confirm"):
                from src.r2_storage import object_exists, get_object_size
                client = _get_r2_client()
                obj_key = pending["object_key"]
                try:
                    if object_exists(client, _r2_bucket, obj_key):
                        size_bytes = get_object_size(client, _r2_bucket, obj_key)
                        already_staged = {f["object_key"] for f in st.session_state["r2_files"]}
                        if obj_key not in already_staged:
                            st.session_state["r2_files"].append({
                                "object_key": obj_key,
                                "filename":   pending["filename"],
                                "size_bytes": size_bytes,
                            })
                        st.session_state["pending_upload"] = None
                        st.rerun()
                    else:
                        st.error(
                            "File not found in R2 yet. Use the Upload button above to "
                            "upload the file first, then click Confirm."
                        )
                except Exception as e:
                    st.error(f"Could not verify upload: {e}")

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
                delete_object(client, _r2_bucket, r2f["object_key"])
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
    and _api_key
)

if not claimant_name.strip():
    st.info("Enter the claimant's name to continue.")
elif not files_ready_for_processing:
    st.info("Upload at least one PDF file to continue.")
elif not _api_key:
    st.warning("ANTHROPIC_API_KEY environment variable is not set.")

# ---------------------------------------------------------------------------
# Past runs history
# ---------------------------------------------------------------------------

_past_sessions = list_sessions()
if _past_sessions:
    with st.expander(f"Past runs ({len(_past_sessions)})"):
        for session in _past_sessions:
            col_claimant, col_file, col_date, col_count, col_dl, col_del = st.columns(
                [2, 2, 2, 1, 1, 1]
            )
            col_claimant.write(session["claimant"])
            col_file.write(session["filenames"])
            col_date.write(session["created_at"][:16].replace("T", " ") + " UTC")
            col_count.write(str(session["report_count"]))

            sid       = session["session_id"]
            dl_label  = f"dl_{sid}"
            del_label = f"del_{sid}"

            try:
                session_summaries = get_session_summaries(sid)
                docx_bytes        = generate_word_document(session["claimant"], session_summaries)
                safe              = session["claimant"].replace(" ", "_")
                col_dl.download_button(
                    label="⬇",
                    data=docx_bytes,
                    file_name=f"{safe}_medical_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=dl_label,
                    help="Download Word document",
                )
            except Exception:
                col_dl.write("—")

            if col_del.button("🗑", key=del_label, help="Delete this run"):
                try:
                    delete_session(sid)
                except Exception as e:
                    st.error(f"Could not delete run: {e}")
                st.rerun()

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if st.button("Process documents", disabled=not ready, type="primary"):
    all_summaries = []
    skipped_reports = []   # list of dicts: {reason, title, start_page, end_page, file}
    session_id = _uuid.uuid4().hex

    def make_skip_callback(f_label):
        def _on_skipped(reason, title, start_page, end_page):
            skipped_reports.append({
                "reason":     reason,
                "title":      title,
                "start_page": start_page,
                "end_page":   end_page,
                "file":       f_label,
            })
        return _on_skipped

    if r2_configured:
        from src.r2_storage import download_as_bytes
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
    status_box  = st.empty()

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
                try:
                    client_r2 = _get_r2_client()
                    with st.spinner(f"Downloading {file_label} from R2…"):
                        pdf_source = download_as_bytes(client_r2, _r2_bucket, source_ref)
                except Exception as e:
                    st.error(f"Failed to download **{file_label}** from R2: {e}")
                    continue
            else:
                pdf_source = source_ref.read()

            file_summaries = process_pdf(
                pdf_source=pdf_source,
                summary_prompt=summary_prompt,
                api_key=_api_key,
                output_path=os.devnull,
                progress_file=progress_file,
                ocr_backend=ocr_backend,
                analysis_model=analysis_model,
                claimant_name=claimant_name.strip(),
                on_progress=make_progress_callback(file_idx, file_label, total_files),
                on_skipped=make_skip_callback(file_label),
            )

            offset = len(all_summaries)
            for s in file_summaries:
                s.report_index = offset + s.report_index

            all_summaries.extend(file_summaries)
            new_count = len(all_summaries) - file_summaries_before
            status_box.markdown(
                f"**{file_label}** — done. {new_count} relevant report(s) found."
            )

            if file_summaries:
                try:
                    save_run(claimant_name.strip(), file_label, file_summaries, session_id=session_id)
                except Exception as e:
                    st.warning(f"Could not save results to database: {e}")

        except anthropic.AuthenticationError:
            st.error(
                "Invalid Anthropic API key — check the ANTHROPIC_API_KEY environment variable."
            )
            break

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

    overall_bar.progress(1.0, text="Done.")

    if not all_summaries:
        st.warning(
            "No relevant medical reports were found in the uploaded file(s). "
            "All detected documents were filtered out by the relevance check."
        )
    else:
        st.success(f"Found **{len(all_summaries)}** relevant report(s).")

        if skipped_reports:
            with st.expander(f"Skipped / filtered reports ({len(skipped_reports)})"):
                for sk in skipped_reports:
                    reason_label = {
                        "not_relevant": "Not relevant",
                        "no_text":      "No OCR text",
                        "duplicate":    "Duplicate",
                    }.get(sk["reason"], sk["reason"])
                    st.write(
                        f"**{reason_label}** — {sk['title']} "
                        f"(pages {sk['start_page']}–{sk['end_page']}, file: {sk['file']})"
                    )

        st.subheader("Summaries")
        for s in all_summaries:
            with st.expander(f"[{s.report_index}] {s.title}  (pages {s.start_page}–{s.end_page})"):
                st.write(s.summary)

        st.subheader("Download")

        docx_bytes = generate_word_document(claimant_name.strip(), all_summaries)
        safe_name  = claimant_name.strip().replace(" ", "_")
        st.download_button(
            label="Download Word document (.docx)",
            data=docx_bytes,
            file_name=f"{safe_name}_medical_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
