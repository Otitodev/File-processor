# File-processor — Medical PDF Report Analyzer

Analyzes large scanned PDFs that contain multiple concatenated medical reports.
Produces a list of every relevant report found, with an AI-generated summary of each one,
and exports the results as a formatted Word document.

Handles files of **any size** (thousands of pages) by processing in small batches
and saving progress after each batch so interrupted runs can be resumed.

Two interfaces are available — a **Streamlit web UI** for interactive use and a
**CLI** for scripted or automated workflows.

---

## How it works

```
     ┌──────────────────┐         ┌─────────────────────┐
     │  CLI  (main.py)  │         │  Web UI  (app.py)   │
     └────────┬─────────┘         └──────────┬──────────┘
              └──────────────┬───────────────┘
                             ↓
              [Page Extraction]    PyMuPDF renders pages as images
                             ↓
              [OCR]               Tesseract (free) or Claude Vision
                             ↓
              [Boundary Detection] Claude identifies where each report starts/ends
                             ↓
              [Relevance Filter]   Claude drops reports unrelated to the claimant
                             ↓
              [Summarization]      Claude summarizes each relevant report
                             ↓
             ┌──────────────┴──────────────┐
             │  JSON file  (CLI)            │
             │  .docx download  (Web UI)    │
             └─────────────────────────────┘
```

The PDF is never fully loaded into memory. Pages are processed in configurable
batches (default: 30 pages). Overlapping pages between batches ensure report
boundaries that fall between chunks are never missed.

---

## Choosing an OCR backend

The biggest cost/quality decision is which OCR engine to use. Two backends are
supported and selectable in the web UI or with `--ocr-backend` on the CLI.

### Backend comparison

| Backend | Cost | Quality | Setup |
|---|---|---|---|
| `tesseract` (default) | **Free** (local) | Good for clean scans | Install system package |
| `claude` | ~$12–24 per 2,000 pages | Best (handles poor scans) | Anthropic API key |

> **Recommendation for 2,000-page files:** Start with Tesseract. OCR cost is $0
> and you only pay Claude for boundary detection, filtering, and summarization (~$3–8 total).
> Switch to `--ocr-backend claude` only if Tesseract quality is poor on your scans.

### Estimated total cost for 2,000 pages

| OCR backend | OCR cost | Claude analysis + summaries | Total |
|---|---|---|---|
| Tesseract (default) | $0 | ~$3–8 | **~$3–8** |
| Claude Vision (`claude-haiku-4-5`) | ~$24 | ~$3–8 | ~$27–32 |

Claude Vision OCR uses `claude-haiku-4-5` (cheapest vision-capable model).
Boundary detection, relevance filtering, and summarization use the model set in
the web UI or via `--model` on the CLI (default: `claude-sonnet-4-6`). Switch to
`claude-haiku-4-5-20251001` to reduce analysis costs further.

### Quick decision guide

```
Are the scans clean (typed text, not handwritten or heavily degraded)?
├── Yes → Use Tesseract (default). Total cost ~$3–8. No extra setup.
└── No  → Is this a one-off job?
           ├── Yes → --ocr-backend claude. Clean results, ~$27–32 for 2k pages.
           └── No  → Consider adding AWS Textract as a third backend (~$3 OCR
                      for 2k pages). Open an issue or PR to contribute it.
```

---

## Setup

### 1. Install system dependencies

```bash
# Tesseract OCR (required for the default "tesseract" OCR backend)
sudo apt-get install tesseract-ocr   # Debian/Ubuntu
brew install tesseract               # macOS
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Dependencies include: `anthropic`, `PyMuPDF`, `pytesseract`, `Pillow`, `click`,
`tqdm`, `streamlit`, `python-docx`.

### 3. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Web UI (Streamlit)

The web UI is the recommended interface for interactive use. No command-line flags
are needed — everything is configured through a browser form.

### Start the app

```bash
streamlit run app.py
```

This opens `http://localhost:8501` in your browser.

### Form fields

| Field | Description |
|---|---|
| **Claimant name** | Full name of the claimant. Used in prompts and by the relevance filter to skip unrelated reports. |
| **Upload PDF file(s)** | One or more scanned PDF disclosure packages. |
| **Summarization prompt** | The instruction sent to Claude for each report. Pre-filled with a default Ontario auto insurance prompt — edit freely. |
| **OCR backend** | `tesseract` (free, local) or `claude` (Claude Vision, higher accuracy). |
| **Claude model** | Model for boundary detection, relevance filtering, and summarization. |
| **Anthropic API key** | Falls back to the `ANTHROPIC_API_KEY` environment variable if set. |

### Processing and output

Click **Process documents**. A progress bar updates in real time as each file
moves through OCR → boundary detection → relevance filtering → summarization.

When complete:
- Each relevant report appears in a collapsible expander showing its title, page range, and summary.
- A **Download Word document (.docx)** button exports all summaries as a formatted
  Word file named `<claimant>_medical_summary.docx`.

---

## Relevance filtering

After boundary detection, Claude evaluates each extracted report against the
claimant's name to decide whether it is relevant to the claim file. Reports that
are unrelated (e.g., a different patient, administrative pages, blank separators)
are silently skipped and do not appear in the output or the Word document.

This step is performed automatically. Provide the claimant name in the web UI
field or via `--claimant-name` on the CLI.

---

## CLI

```bash
python main.py <PDF_FILE> --prompt "<your summarization prompt>"
```

### Examples

**Basic usage (Ontario auto insurance disclosure package):**
```bash
python main.py disclosure_package.pdf \
  --prompt "Summarize this medical document for use in an Ontario auto insurance claim file."
```

**With relevance filtering:**
```bash
python main.py patient_records.pdf \
  --prompt "Summarize this medical report." \
  --claimant-name "Jane Smith"
```

**High-accuracy OCR** (better for poor-quality scans, uses Claude Vision):
```bash
python main.py disclosure_package.pdf \
  --prompt "Summarize this medical document for use in an Ontario auto insurance claim file." \
  --ocr-backend claude
```

**Cheaper analysis model** (reduces Claude API cost):
```bash
python main.py patient_records.pdf \
  --prompt "Summarize this medical report." \
  --model claude-haiku-4-5-20251001
```

**Custom batch size** (reduce if running out of memory):
```bash
python main.py large_file.pdf \
  --prompt "..." \
  --batch-size 15
```

**Resume an interrupted run** — just re-run the same command. Progress is
automatically restored from `.pipeline_progress.json`. Delete that file to
start fresh.

---

## Output

### JSON (CLI)

Results are saved to `results.json` (configurable with `--output`):

```json
[
  {
    "report_index": 1,
    "title": "INSURER'S EXAMINATION – MEDICAL PHYSICIAN ASSESSMENT — Dr. Mohamed Khaled",
    "start_page": 1,
    "end_page": 4,
    "summary": "Dr. Mohamed Khaled, Physician, completed an INSURER'S EXAMINATION – MEDICAL PHYSICIAN ASSESSMENT dated December 30, 2025, opining that the claimant sustained significant accident-related spinal injuries, including a comminuted burst fracture of the T12 vertebral body with central canal stenosis and a cervical vertebral fracture requiring surgical intervention. Dr. Khaled documented ongoing axial spinal pain aggravated by activity and prolonged positioning, the use of a four-wheel walker and back brace for ambulation, and structural range of motion impairments with functional impairment related to muscular deconditioning and cervical and/or lumbar sprain/strain. Dr. Khaled identified no pre-accident conditions from a physical medicine perspective.\n\nWith respect to the disputed OCF-18 dated October 24, 2025, in the amount of $2,200.00, Dr. Khaled opined that the proposed goods and services were reasonable and necessary as a direct result of the accident-related injuries. The plan included assessment services totaling $2,200.00, consisting of a total body assessment and documentation support activity."
  },
  {
    "report_index": 2,
    "title": "OCF-18 Treatment and Assessment Plan — Laura Nelson",
    "start_page": 5,
    "end_page": 7,
    "summary": "Laura Nelson, Occupational Therapist (College Registration Number G1911702), completed an OCF-18 Treatment and Assessment Plan (Effective date 2016-10-01) dated September 9, 2025, proposing occupational therapy services totaling $3,940.04. The proposed services included therapy for motor and living skills, provider travel time, documentation support activity, and brokerage/service fees. The document reflects approval of the amount of $3,940.04."
  }
]
```

Each `summary` is written as professional prose paragraph(s) suitable for an insurance claim file. The format covers:

- **Author identification**: name, credential, and registration number where present
- **Document type and date**: exact document title and date
- **Clinical findings**: injuries, diagnoses, functional limitations, and equipment/aids documented
- **Treatment plans (OCF-18)**: proposed services, dollar amounts, and approval status
- **Disability certificates (OCF-3)**: listed injuries and functional inability statements

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--prompt` / `-p` | *(required)* | Summarization prompt applied to each report |
| `--claimant-name` | *(empty)* | Claimant's full name; injected into prompts and used by the relevance filter |
| `--api-key` / `-k` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--output` / `-o` | `results.json` | Output file path |
| `--batch-size` / `-b` | `30` | Pages per batch (lower = less RAM) |
| `--overlap` | `2` | Overlap pages between batches (prevents missing boundaries) |
| `--dpi` | `200` | Rendering DPI for OCR (200 is a good default) |
| `--ocr-backend` | `tesseract` | `tesseract` (free, local) or `claude` (higher accuracy, uses image tokens) |
| `--model` | `claude-sonnet-4-6` | Claude model for boundary detection, filtering, and summarization |
| `--progress-file` | `.pipeline_progress.json` | Checkpoint file for resuming |

---

## Project structure

```
File-processor/
├── app.py               # Streamlit web UI entry point
├── main.py              # CLI entry point
├── requirements.txt
└── src/
    ├── __init__.py
    ├── extractor.py        # PDF → page images (PyMuPDF)
    ├── ocr_engine.py       # Page images → text (Tesseract or Claude Vision)
    ├── report_analyzer.py  # Boundary detection, relevance filtering, summarization (Claude)
    ├── docx_writer.py      # ReportSummary list → formatted .docx bytes
    └── pipeline.py         # Orchestrates full pipeline with checkpointing
```
