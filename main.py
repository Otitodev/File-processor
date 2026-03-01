#!/usr/bin/env python3
"""
CLI entry point for the medical PDF report analyzer.

Usage examples:

  # Insurance/legal medical document analysis (standard use case)
  python main.py disclosure_package.pdf \
      --prompt "Summarize this medical document for use in an Ontario auto insurance claim file." \
      --api-key sk-ant-...

  # High-accuracy mode — Claude Vision OCR (better for poor-quality scans)
  python main.py disclosure_package.pdf \
      --prompt "Summarize this medical document for use in an Ontario auto insurance claim file." \
      --api-key sk-ant-... \
      --ocr-backend claude

  # Custom batch size and output file
  python main.py large_file.pdf \
      --prompt "Summarize this medical document for use in an Ontario auto insurance claim file." \
      --api-key sk-ant-... \
      --batch-size 20 \
      --output my_results.json

  # Resume an interrupted run (just re-run the same command)
  python main.py disclosure_package.pdf \
      --prompt "..." \
      --api-key sk-ant-...
      # Progress is automatically restored from .pipeline_progress.json
"""

import os
import sys

import click

from src.pipeline import process_pdf


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, readable=True))
@click.option(
    "--prompt", "-p",
    required=True,
    help="The summarization prompt to apply to each report.",
)
@click.option(
    "--api-key", "-k",
    default=lambda: os.environ.get("ANTHROPIC_API_KEY", ""),
    show_default="ANTHROPIC_API_KEY env var",
    help="Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable.",
)
@click.option(
    "--output", "-o",
    default="results.json",
    show_default=True,
    help="Path for the JSON output file.",
)
@click.option(
    "--batch-size", "-b",
    default=30,
    show_default=True,
    help="Number of PDF pages to process per batch. Lower = less RAM.",
)
@click.option(
    "--overlap",
    default=2,
    show_default=True,
    help="Pages repeated between batches so report boundaries are not missed.",
)
@click.option(
    "--dpi",
    default=200,
    show_default=True,
    help="Rendering DPI for page images. 200 is good for OCR; lower saves memory.",
)
@click.option(
    "--ocr-backend",
    type=click.Choice(["tesseract", "claude"], case_sensitive=False),
    default="tesseract",
    show_default=True,
    help=(
        "OCR engine. 'tesseract' is free and local. "
        "'claude' uses Claude Vision (more accurate, uses image tokens)."
    ),
)
@click.option(
    "--model",
    default="claude-sonnet-4-6",
    show_default=True,
    help="Claude model for boundary detection and summarization.",
)
@click.option(
    "--progress-file",
    default=".pipeline_progress.json",
    show_default=True,
    help="Checkpoint file. Delete this file to start fresh instead of resuming.",
)
@click.option(
    "--claimant-name", "-n",
    default="",
    help="Claimant's full name. Injected into prompts for context and relevance filtering.",
)
def main(
    pdf_path,
    prompt,
    api_key,
    output,
    batch_size,
    overlap,
    dpi,
    ocr_backend,
    model,
    progress_file,
    claimant_name,
):
    """
    Analyze a large scanned PDF containing multiple medical reports.

    Extracts each report, identifies its boundaries, and produces a summary
    using your custom PROMPT. Results are saved as a JSON array to OUTPUT.

    The run can be interrupted and resumed at any time — progress is
    automatically saved after each batch.
    """
    if not api_key:
        click.echo(
            "Error: No API key provided. Set ANTHROPIC_API_KEY or use --api-key.",
            err=True,
        )
        sys.exit(1)

    summaries = process_pdf(
        pdf_path=pdf_path,
        summary_prompt=prompt,
        api_key=api_key,
        output_path=output,
        progress_file=progress_file,
        batch_size=batch_size,
        overlap=overlap,
        dpi=dpi,
        ocr_backend=ocr_backend,
        analysis_model=model,
        claimant_name=claimant_name,
    )

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Reports found: {len(summaries)}")
    click.echo(f"{'=' * 60}")
    for s in summaries:
        click.echo(f"\n[{s.report_index}] {s.title} (pages {s.start_page}–{s.end_page})")
        click.echo(f"    {s.summary[:200]}{'...' if len(s.summary) > 200 else ''}")


if __name__ == "__main__":
    main()
