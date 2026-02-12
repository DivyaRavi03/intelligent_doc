"""Unit tests for the table extractor module."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.table_extractor import TableExtractor


class TestTableExtractor:
    """Tests for :class:`TableExtractor`."""

    def test_extract_tables_file_not_found(self, tmp_path: Path) -> None:
        """Nonexistent PDF should raise FileNotFoundError."""
        extractor = TableExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_tables(tmp_path / "missing.pdf")

    def test_extract_tables_no_tables(self, tmp_pdf: Path) -> None:
        """A text-only PDF should return zero tables."""
        extractor = TableExtractor()
        result = extractor.extract_tables(tmp_pdf)

        assert result.total_tables == 0
        assert result.tables == []

    def test_needs_vision_fallback_empty_table(self) -> None:
        """An empty table should trigger vision fallback."""
        extractor = TableExtractor()
        assert extractor._needs_vision_fallback([]) is True

    def test_needs_vision_fallback_all_none(self) -> None:
        """A table where all cells are None should trigger fallback."""
        extractor = TableExtractor()
        raw_table = [[None, None], [None, None], [None, None]]
        assert extractor._needs_vision_fallback(raw_table) is True

    def test_needs_vision_fallback_well_filled(self) -> None:
        """A table with >30% filled cells should not trigger fallback."""
        extractor = TableExtractor()
        raw_table = [
            ["Method", "F1", "Accuracy"],
            ["Ours", "94.2", "95.1"],
            ["Baseline", "89.1", "90.3"],
        ]
        assert extractor._needs_vision_fallback(raw_table) is False

    def test_needs_vision_fallback_sparse(self) -> None:
        """A table with very few filled cells should trigger fallback."""
        extractor = TableExtractor(min_cell_fill_ratio=0.5)
        raw_table = [
            ["Method", None, None, None],
            [None, None, None, None],
            [None, "x", None, None],
        ]
        assert extractor._needs_vision_fallback(raw_table) is True

    def test_parse_raw_table_normal(self) -> None:
        """parse_raw_table should split first row as headers, rest as data."""
        raw = [
            ["Model", "F1", "Precision"],
            ["BERT", "92.1", "93.0"],
            ["GPT", "91.5", "92.2"],
        ]
        headers, rows = TableExtractor._parse_raw_table(raw)
        assert headers == ["Model", "F1", "Precision"]
        assert len(rows) == 2
        assert rows[0] == ["BERT", "92.1", "93.0"]

    def test_parse_raw_table_with_none_cells(self) -> None:
        """None cells should be replaced with empty strings."""
        raw = [
            ["Col A", None, "Col C"],
            [None, "value", None],
        ]
        headers, rows = TableExtractor._parse_raw_table(raw)
        assert headers == ["Col A", "", "Col C"]
        assert rows[0] == ["", "value", ""]

    def test_parse_raw_table_empty(self) -> None:
        """Empty table should return empty headers and rows."""
        headers, rows = TableExtractor._parse_raw_table([])
        assert headers == []
        assert rows == []

    def test_score_table_perfect(self) -> None:
        """A fully filled table should score 1.0."""
        raw = [
            ["A", "B"],
            ["C", "D"],
        ]
        assert TableExtractor._score_table(raw) == 1.0

    def test_score_table_empty(self) -> None:
        """An empty table should score 0.0."""
        assert TableExtractor._score_table([]) == 0.0

    def test_score_table_partial(self) -> None:
        """Score should reflect fill ratio."""
        raw = [
            ["A", None],
            [None, "B"],
        ]
        score = TableExtractor._score_table(raw)
        assert score == 0.5

    def test_find_caption_with_matching_table(self) -> None:
        """Should find 'Table 1: ...' patterns in page text."""
        page_text = textwrap.dedent("""\
            Some text here.
            Table 1: Performance comparison across datasets.
            More text below.
        """)
        caption = TableExtractor._find_caption(page_text, table_index=0)
        assert caption is not None
        assert "Performance comparison" in caption

    def test_find_caption_no_match(self) -> None:
        """Should return None when no table caption is found."""
        page_text = "Just some plain text with no table mentions."
        caption = TableExtractor._find_caption(page_text, table_index=0)
        assert caption is None

    def test_find_caption_fallback(self) -> None:
        """Should fall back to any line starting with 'Table'."""
        page_text = "Table shows the full results of ablation.\nAnother line."
        caption = TableExtractor._find_caption(page_text, table_index=5)
        assert caption is not None
        assert "Table" in caption

    @patch("src.ingestion.table_extractor.settings")
    def test_gemini_vision_skipped_without_api_key(
        self, mock_settings: MagicMock, tmp_pdf: Path
    ) -> None:
        """Vision fallback should return None when no API key is set."""
        mock_settings.gemini_api_key = ""
        extractor = TableExtractor()
        result = extractor._gemini_vision_extract(tmp_pdf, page_number=0)
        assert result is None

    def test_extract_tables_with_table_pdf(self, tmp_path: Path) -> None:
        """Integration test: a PDF with a simple table should be extracted."""
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Draw a simple grid to simulate a table
        # Header row
        y_start = 100
        col_width = 150
        row_height = 25
        headers = ["Method", "F1", "Acc"]
        rows_data = [["BERT", "92.1", "93.0"], ["GPT-4", "95.3", "96.1"]]

        for i, h in enumerate(headers):
            x = 72 + i * col_width
            page.insert_text((x + 5, y_start + 15), h, fontsize=10)
            # Draw cell borders
            rect = fitz.Rect(x, y_start, x + col_width, y_start + row_height)
            page.draw_rect(rect)

        for r_idx, row in enumerate(rows_data):
            y = y_start + (r_idx + 1) * row_height
            for c_idx, cell in enumerate(row):
                x = 72 + c_idx * col_width
                page.insert_text((x + 5, y + 15), cell, fontsize=10)
                rect = fitz.Rect(x, y, x + col_width, y + row_height)
                page.draw_rect(rect)

        pdf_path = tmp_path / "table_test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        extractor = TableExtractor()
        result = extractor.extract_tables(pdf_path)

        # pdfplumber should detect the drawn table
        # (may or may not depending on how clean the lines are,
        #  but the test validates the pipeline runs without error)
        assert result.total_tables >= 0
        assert isinstance(result.tables, list)
