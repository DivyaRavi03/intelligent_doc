"""PDF text extraction with native (PyMuPDF) and OCR (Tesseract) fallback.

The :class:`PDFParser` processes each page independently, choosing native
extraction when text is present and falling back to OCR for scanned pages.
A per-page confidence score reflects extraction quality.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.models.schemas import ExtractionMethod, PageResult, PDFExtractionResult

logger = logging.getLogger(__name__)

# Minimum ratio of printable characters for a page to be considered "native"
_NATIVE_TEXT_THRESHOLD = 0.3
# Minimum absolute character count for native extraction
_MIN_CHAR_COUNT = 20
# DPI for rendering pages to images for OCR
_OCR_DPI = 300


class PDFParser:
    """Extract text from PDF files using native text or OCR.

    Args:
        ocr_dpi: Resolution for rendering pages for OCR.
        native_threshold: Minimum text-density ratio to skip OCR.
    """

    def __init__(
        self,
        ocr_dpi: int = _OCR_DPI,
        native_threshold: float = _NATIVE_TEXT_THRESHOLD,
    ) -> None:
        self.ocr_dpi = ocr_dpi
        self.native_threshold = native_threshold

    def extract(self, pdf_path: str | Path) -> PDFExtractionResult:
        """Extract text from all pages of a PDF.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A :class:`PDFExtractionResult` with per-page text and scores.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            RuntimeError: If the PDF cannot be opened.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF: {exc}") from exc

        pages: list[PageResult] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                if self._needs_ocr(page):
                    result = self._extract_ocr_page(page, page_num)
                else:
                    result = self._extract_native_page(page, page_num)
                pages.append(result)
        finally:
            doc.close()

        total = len(pages)
        avg_conf = sum(p.confidence for p in pages) / total if total else 0.0

        return PDFExtractionResult(
            pages=pages,
            total_pages=total,
            avg_confidence=round(avg_conf, 4),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _needs_ocr(self, page: fitz.Page) -> bool:
        """Decide whether a page requires OCR.

        A page needs OCR when its native text is too sparse relative to the
        page area, suggesting the content is a scanned image.
        """
        text = page.get_text("text") or ""
        if len(text.strip()) < _MIN_CHAR_COUNT:
            return True

        # Ratio of text blocks area to page area
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]
        if not text_blocks:
            return True

        total_chars = sum(
            len(span.get("text", ""))
            for block in text_blocks
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        # Rough heuristic: if chars per block is too low, it might be artefacts
        chars_per_block = total_chars / len(text_blocks) if text_blocks else 0
        return chars_per_block < self.native_threshold * 100

    def _extract_native_page(self, page: fitz.Page, page_number: int) -> PageResult:
        """Extract text natively via PyMuPDF."""
        raw_text = page.get_text("text") or ""
        cleaned = self._clean_text(raw_text)
        confidence = self._compute_confidence(page, cleaned, method="native")

        return PageResult(
            page_number=page_number,
            text=cleaned,
            extraction_method=ExtractionMethod.NATIVE,
            confidence=round(confidence, 4),
            char_count=len(cleaned),
        )

    def _extract_ocr_page(self, page: fitz.Page, page_number: int) -> PageResult:
        """Render the page to an image and run Tesseract OCR."""
        import pytesseract

        pix = page.get_pixmap(dpi=self.ocr_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        raw_text: str = pytesseract.image_to_string(img)
        cleaned = self._clean_text(raw_text)
        confidence = self._compute_confidence(page, cleaned, method="ocr")

        return PageResult(
            page_number=page_number,
            text=cleaned,
            extraction_method=ExtractionMethod.OCR,
            confidence=round(confidence, 4),
            char_count=len(cleaned),
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise extracted text.

        - Collapse multiple blank lines into a single one.
        - Strip trailing whitespace per line.
        - Remove non-printable control characters (except newline/tab).
        """
        # Remove control chars except \n \t
        text = re.sub(r"[^\S \n\t]+", " ", text)
        # Collapse runs of whitespace-only lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing spaces on each line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _compute_confidence(page: fitz.Page, text: str, method: str) -> float:
        """Heuristic confidence score in [0, 1].

        For native extraction we start high and penalise sparse text.
        For OCR we start lower and reward density.
        """
        char_count = len(text)
        page_area = page.rect.width * page.rect.height

        if page_area == 0:
            return 0.0

        density = char_count / (page_area / 1000)  # chars per 1k area units

        if method == "native":
            # High baseline; penalise if text is very short
            score = min(1.0, 0.85 + density * 0.01)
        else:
            # OCR starts lower
            score = min(1.0, 0.5 + density * 0.02)

        return max(0.0, min(score, 1.0))
