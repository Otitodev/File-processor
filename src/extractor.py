"""
PDF page extractor — converts PDF pages to PIL Images using PyMuPDF.
Supports very large PDFs by yielding one page at a time.
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import Generator, Tuple, Union


def _open_doc(pdf_source: Union[str, bytes]) -> fitz.Document:
    """Open a PDF from a file path or raw bytes."""
    if isinstance(pdf_source, bytes):
        return fitz.open(stream=pdf_source, filetype="pdf")
    return fitz.open(pdf_source)


def iter_pages(pdf_source: Union[str, bytes], dpi: int = 200) -> Generator[Tuple[int, Image.Image], None, None]:
    """
    Yield (page_number, PIL.Image) for every page in the PDF.

    Args:
        pdf_source: Path to the PDF file, or raw PDF bytes.
        dpi: Rendering resolution. 200 dpi is good for OCR; lower saves memory.

    Yields:
        (1-based page number, PIL Image in RGB mode)
    """
    doc = _open_doc(pdf_source)
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is PyMuPDF's base DPI
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            yield page_num + 1, img
    finally:
        doc.close()


def get_page_count(pdf_source: Union[str, bytes]) -> int:
    """Return the total number of pages without loading all of them."""
    doc = _open_doc(pdf_source)
    count = len(doc)
    doc.close()
    return count


def iter_page_batches(
    pdf_source: Union[str, bytes],
    batch_size: int = 30,
    dpi: int = 200,
    overlap: int = 2,
) -> Generator[Tuple[int, int, list], None, None]:
    """
    Yield (start_page, end_page, [PIL Images]) in batches.

    Includes `overlap` pages from the previous batch at the start of the next
    batch so that report boundaries falling between batches are not missed.

    Args:
        pdf_source: Path to the PDF, or raw PDF bytes.
        batch_size: Number of pages per batch.
        dpi:        Rendering DPI.
        overlap:    Pages to repeat between batches for boundary continuity.

    Yields:
        (start_page, end_page, list_of_images)  — page numbers are 1-based.
    """
    doc = _open_doc(pdf_source)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    total = len(doc)

    try:
        start = 0
        while start < total:
            end = min(start + batch_size, total)
            images = []
            for idx in range(start, end):
                page = doc[idx]
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)

            yield start + 1, end, images  # 1-based

            # Move forward, keeping `overlap` pages for the next batch
            start = end - overlap if end < total else total
    finally:
        doc.close()
