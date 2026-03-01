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
[Page Extraction]     PyMuPDF renders pages as images
       ↓
[OCR]                 Tesseract (free, local) or Claude Vision converts images to text
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

## Choosing an OCR backend

The biggest cost/quality decision is which OCR engine to use. Two backends are
supported and selectable with `--ocr-backend`.

### Backend comparison

| Backend | Cost | Quality | Setup |
|---|---|---|---|
| `tesseract` (default) | **Free** (local) | Good for clean scans | Install system package |
| `claude` | ~$12–24 per 2,000 pages | Best (handles poor scans) | Anthropic API key |

> **Recommendation for 2,000-page files:** Start with Tesseract. OCR cost is $0
> and you only pay Claude for boundary detection and summarization (~$3–8 total).
> Switch to `--ocr-backend claude` only if Tesseract quality is poor on your scans.

### Estimated total cost for 2,000 pages

| OCR backend | OCR cost | Claude analysis + summaries | Total |
|---|---|---|---|
| Tesseract (default) | $0 | ~$3–8 | **~$3–8** |
| Claude Vision (`claude-haiku-4-5`) | ~$24 | ~$3–8 | ~$27–32 |

Claude Vision OCR uses `claude-haiku-4-5` (the cheapest vision-capable model).
Boundary detection and summarization always use the model set by `--model`
(default: `claude-sonnet-4-6`). Switch to `claude-haiku-4-5-20251001` via
`--model` to reduce analysis costs further.

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

**Basic usage (Tesseract OCR — free, recommended starting point):**
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

**Cheaper analysis model** (reduces Claude API cost for boundary detection and summarization):
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
| `--ocr-backend` | `tesseract` | `tesseract` (free, local) or `claude` (higher accuracy, uses image tokens) |
| `--model` | `claude-sonnet-4-6` | Claude model for boundary detection and summarization |
| `--progress-file` | `.pipeline_progress.json` | Checkpoint file for resuming |

---

## Project structure

```
File-processor/
├── main.py              # CLI entry point
├── requirements.txt
└── src/
    ├── __init__.py
    ├── extractor.py        # PDF → page images (PyMuPDF)
    ├── ocr_engine.py       # Page images → text (Tesseract or Claude Vision)
    ├── report_analyzer.py  # Boundary detection + summarization (Claude)
    └── pipeline.py         # Orchestrates the full pipeline with checkpointing
```
