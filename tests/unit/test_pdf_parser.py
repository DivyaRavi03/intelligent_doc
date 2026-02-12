"""Unit tests for the PDF parser module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_parser import PDFParser
from src.models.schemas import ExtractionMethod


class TestPDFParser:
    """Tests for :class:`PDFParser`."""

    def test_extract_returns_all_pages(self, tmp_pdf: Path) -> None:
        """Parser should return a result for every page in the PDF."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 0
        assert result.pages[1].page_number == 1

    def test_extract_contains_expected_text(self, tmp_pdf: Path) -> None:
        """Known text inserted into the test PDF should be present in output."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        full_text = " ".join(p.text for p in result.pages)
        assert "Deep Learning for Document Understanding" in full_text
        assert "Introduction" in full_text

    def test_confidence_is_valid_range(self, tmp_pdf: Path) -> None:
        """All confidence scores should be in [0, 1]."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        for page in result.pages:
            assert 0.0 <= page.confidence <= 1.0
        assert 0.0 <= result.avg_confidence <= 1.0

    def test_native_extraction_method(self, tmp_pdf: Path) -> None:
        """Pages with embedded text should use native extraction."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        # Our test PDF has native text, so all pages should be native
        for page in result.pages:
            assert page.extraction_method == ExtractionMethod.NATIVE

    def test_char_count_matches_text_length(self, tmp_pdf: Path) -> None:
        """char_count field should match actual text length."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        for page in result.pages:
            assert page.char_count == len(page.text)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Extracting from a nonexistent path should raise FileNotFoundError."""
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.extract(tmp_path / "nonexistent.pdf")

    def test_invalid_pdf_raises_runtime_error(self, tmp_path: Path) -> None:
        """A corrupted/non-PDF file should raise RuntimeError."""
        bad_file = tmp_path / "bad.pdf"
        bad_file.write_text("this is not a PDF")

        parser = PDFParser()
        with pytest.raises(RuntimeError, match="Failed to open PDF"):
            parser.extract(bad_file)

    def test_clean_text_collapses_blank_lines(self) -> None:
        """_clean_text should collapse 3+ consecutive blank lines."""
        text = "line1\n\n\n\n\nline2"
        result = PDFParser._clean_text(text)
        assert result == "line1\n\nline2"

    def test_clean_text_strips_trailing_whitespace(self) -> None:
        """_clean_text should strip trailing whitespace per line."""
        text = "hello   \nworld   "
        result = PDFParser._clean_text(text)
        assert result == "hello\nworld"

    def test_clean_text_empty_input(self) -> None:
        """Empty input should return empty string."""
        assert PDFParser._clean_text("") == ""
        assert PDFParser._clean_text("   \n  \n  ") == ""

    def test_needs_ocr_with_rich_text(self, mock_fitz_page: MagicMock) -> None:
        """A page with adequate text should not require OCR."""
        parser = PDFParser()
        # The mock returns enough text, so OCR should not be needed
        assert parser._needs_ocr(mock_fitz_page) is False

    def test_needs_ocr_with_sparse_text(self) -> None:
        """A page with very little text should trigger OCR."""
        parser = PDFParser()

        page = MagicMock()
        page.get_text.side_effect = lambda fmt="text", **kw: (
            {"blocks": []} if fmt == "dict" else "ab"
        )

        assert parser._needs_ocr(page) is True

    @patch("pytesseract.image_to_string", return_value="OCR extracted text")
    def test_ocr_fallback_called(self, mock_ocr: MagicMock, tmp_path: Path) -> None:
        """When native text is sparse, OCR should be invoked."""
        import fitz

        # Create a PDF with an image-only page (no text layer)
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        # Draw a rect instead of text so native extraction is empty
        page.draw_rect(fitz.Rect(10, 10, 190, 190), color=(0, 0, 0))
        pdf_path = tmp_path / "image_only.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = PDFParser()
        result = parser.extract(pdf_path)

        assert result.total_pages == 1
        # Tesseract should have been called
        assert mock_ocr.called

    def test_avg_confidence_calculation(self, tmp_pdf: Path) -> None:
        """avg_confidence should be the mean of page confidences."""
        parser = PDFParser()
        result = parser.extract(tmp_pdf)

        expected = sum(p.confidence for p in result.pages) / len(result.pages)
        assert abs(result.avg_confidence - round(expected, 4)) < 1e-4
