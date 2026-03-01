# File-processor — Medical PDF Report Analyzer

Analyzes large scanned PDFs that contain multiple concatenated medical reports.
Produces a list of every report found, with an AI-generated summary of each one.

Handles files of **any size** (thousands of pages) by processing in small batches
and saving progress after each batch so interrupted runs can be resumed.

---

## How it works

```
Large Scanned PDF
       ↓
[Page Extraction]     PyMuPDF extracts pages as images
       ↓
[Batch OCR]           pytesseract (or Claude Vision) converts images to text
       ↓
[Boundary Detection]  Claude identifies where each report starts and ends
       ↓
[Report Grouping]     Pages belonging to the same report are collected
       ↓
[Summarization]       Claude summarizes each report using your custom prompt
       ↓
[Output]              JSON list of reports + summaries
```

The PDF is never fully loaded into memory. Pages are processed in configurable
batches (default: 30 pages). Overlapping pages between batches ensure report
boundaries that fall between chunks are never missed.

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

### 3. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Usage

```bash
python main.py <PDF_FILE> --prompt "<your summarization prompt>"
```

### Examples

**Basic usage:**
```bash
python main.py patient_records.pdf \
  --prompt "Summarize this medical report: include patient name, date, type of report, key findings, and any recommended follow-up."
```

**High-accuracy OCR** (better for poor-quality scans, uses Claude Vision):
```bash
python main.py patient_records.pdf \
  --prompt "Extract: patient name, date of report, diagnosis, and treatment plan." \
  --ocr-backend claude
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

Results are saved to `results.json` (configurable with `--output`):

```json
[
  {
    "report_index": 1,
    "title": "Chest X-Ray — John Smith",
    "start_page": 1,
    "end_page": 3,
    "summary": "Patient John Smith, DOB 1965-04-12. Chest X-ray performed 2024-01-15. ..."
  },
  {
    "report_index": 2,
    "title": "Blood Panel — Jane Doe",
    "start_page": 4,
    "end_page": 6,
    "summary": "..."
  }
]
```

---

## Options

| Option | Default | Description |
|---|---|---|
| `--prompt` / `-p` | *(required)* | Summarization prompt applied to each report |
| `--api-key` / `-k` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--output` / `-o` | `results.json` | Output file path |
| `--batch-size` / `-b` | `30` | Pages per batch (lower = less RAM) |
| `--overlap` | `2` | Overlap pages between batches (prevents missing boundaries) |
| `--dpi` | `200` | Rendering DPI for OCR (200 is a good default) |
| `--ocr-backend` | `tesseract` | `tesseract` (free, local) or `claude` (higher accuracy) |
| `--model` | `claude-sonnet-4-6` | Claude model for analysis and summarization |
| `--progress-file` | `.pipeline_progress.json` | Checkpoint file for resuming |

---

## Project structure

```
File-processor/
├── main.py              # CLI entry point
├── requirements.txt
└── src/
    ├── __init__.py
    ├── extractor.py     # PDF → page images (PyMuPDF)
    ├── ocr_engine.py    # Page images → text (Tesseract or Claude Vision)
    ├── report_analyzer.py  # Boundary detection + summarization (Claude)
    └── pipeline.py      # Orchestrates the full pipeline with checkpointing
```
