"""
OCR engine — extracts text from page images.

Two backends are supported:
  - "tesseract"  : Local, free, good for clean scans. Requires tesseract-ocr installed.
  - "claude"     : Uses Claude Vision API. Higher accuracy on poor-quality scans,
                   but uses image tokens (more expensive). Falls back to tesseract
                   automatically if ANTHROPIC_API_KEY is not set.
"""

import base64
import io
from typing import List, Literal

from PIL import Image

from .report_analyzer import _create_with_retry

OCRBackend = Literal["tesseract", "claude"]


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def ocr_page_tesseract(img: Image.Image) -> str:
    """Extract text from a single page image using Tesseract."""
    try:
        import pytesseract
    except ImportError as e:
        raise ImportError("pytesseract is required for the 'tesseract' backend. Run: pip install pytesseract") from e

    # --psm 3  : Fully automatic page segmentation (good for multi-column layouts)
    # --oem 3  : Default OCR Engine Mode (neural nets + legacy)
    config = "--psm 3 --oem 3"
    return pytesseract.image_to_string(img, config=config)


def ocr_page_claude(img: Image.Image, client) -> str:
    """Extract text from a single page image using Claude Vision."""
    b64 = _image_to_b64(img)
    response = _create_with_retry(
        client,
        model="claude-haiku-4-5-20251001",  # cheapest vision-capable model
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please transcribe all text from this scanned medical document page. "
                            "Preserve the layout as much as possible using line breaks. "
                            "Output ONLY the transcribed text, no commentary."
                        ),
                    },
                ],
            }
        ],
    )
    return response.content[0].text


def ocr_batch(
    images: List[Image.Image],
    backend: OCRBackend = "tesseract",
    client=None,
) -> List[str]:
    """
    OCR a list of page images, returning a list of extracted text strings.

    Args:
        images:  List of PIL Images (one per page).
        backend: "tesseract" or "claude".
        client:  anthropic.Anthropic client instance (required for "claude" backend).

    Returns:
        List of text strings, one per page.
    """
    results = []
    for img in images:
        if backend == "claude":
            if client is None:
                raise ValueError("An anthropic client must be supplied for the 'claude' OCR backend.")
            text = ocr_page_claude(img, client)
        else:
            text = ocr_page_tesseract(img)
        results.append(text)
    return results
