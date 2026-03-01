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

**Basic usage (Ontario auto insurance disclosure package):**
```bash
python main.py disclosure_package.pdf \
  --prompt "Summarize this medical document for use in an Ontario auto insurance claim file."
```

**High-accuracy OCR** (better for poor-quality scans, uses Claude Vision):
```bash
python main.py disclosure_package.pdf \
  --prompt "Summarize this medical document for use in an Ontario auto insurance claim file." \
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
