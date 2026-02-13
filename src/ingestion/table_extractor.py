"""Table extraction from PDF pages.

Primary extraction uses **pdfplumber** for rule-based table detection.
When tables appear malformed or empty, a **Gemini Vision** fallback renders
the page as an image and asks the model to extract the table in JSON.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.config import settings
from src.models.schemas import ExtractedTable, TableExtractionResult

logger = logging.getLogger(__name__)

# If a pdfplumber table has fewer non-empty cells than this ratio, use vision fallback
_MIN_CELL_FILL_RATIO = 0.3
_VISION_DPI = 200


class TableExtractor:
    """Extract tables from a PDF, with Gemini Vision fallback.

    Args:
        min_cell_fill_ratio: Threshold below which a pdfplumber table is
            considered too noisy and the vision fallback is triggered.
    """

    def __init__(self, min_cell_fill_ratio: float = _MIN_CELL_FILL_RATIO) -> None:
        self.min_cell_fill_ratio = min_cell_fill_ratio

    def extract_tables(self, pdf_path: str | Path) -> TableExtractionResult:
        """Extract all tables from every page of *pdf_path*.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            :class:`TableExtractionResult` containing all detected tables.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        tables: list[ExtractedTable] = []
        page_texts = self._get_page_texts(pdf_path)

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                raw_tables = self._pdfplumber_extract(page)

                for tbl_idx, raw_tbl in enumerate(raw_tables):
                    if self._needs_vision_fallback(raw_tbl):
                        logger.info(
                            "Triggering vision fallback for page %d table %d",
                            page_idx,
                            tbl_idx,
                        )
                        vision_result = self._gemini_vision_extract(pdf_path, page_idx)
                        if vision_result:
                            for vi, vt in enumerate(vision_result):
                                vt.page_number = page_idx
                                vt.table_index = tbl_idx + vi
                                vt.caption = self._find_caption(
                                    page_texts.get(page_idx, ""), tbl_idx + vi
                                )
                                tables.append(vt)
                            break  # vision fallback already handled the page
                    else:
                        headers, rows = self._parse_raw_table(raw_tbl)
                        caption = self._find_caption(page_texts.get(page_idx, ""), tbl_idx)
                        tables.append(
                            ExtractedTable(
                                page_number=page_idx,
                                table_index=tbl_idx,
                                headers=headers,
                                rows=rows,
                                caption=caption,
                                extraction_method="pdfplumber",
                                confidence=self._score_table(raw_tbl),
                            )
                        )

        return TableExtractionResult(tables=tables, total_tables=len(tables))

    # ------------------------------------------------------------------
    # pdfplumber extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _pdfplumber_extract(page: pdfplumber.page.Page) -> list[list[list[str | None]]]:
        """Run pdfplumber table detection on a single page.

        Returns a list of tables, each table being a list of rows,
        each row a list of cell values (possibly ``None``).
        """
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
        }
        try:
            tables = page.extract_tables(table_settings)
            return tables if tables else []
        except Exception:
            logger.warning("pdfplumber extraction failed, trying text strategy", exc_info=True)
            table_settings["vertical_strategy"] = "text"
            table_settings["horizontal_strategy"] = "text"
            try:
                tables = page.extract_tables(table_settings)
                return tables if tables else []
            except Exception:
                logger.error("pdfplumber extraction failed completely", exc_info=True)
                return []

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------

    def _needs_vision_fallback(self, raw_table: list[list[str | None]]) -> bool:
        """Determine whether the pdfplumber result is too low quality."""
        if not raw_table:
            return True

        total_cells = sum(len(row) for row in raw_table)
        if total_cells == 0:
            return True

        filled = sum(1 for row in raw_table for cell in row if cell and cell.strip())
        return (filled / total_cells) < self.min_cell_fill_ratio

    # ------------------------------------------------------------------
    # Gemini Vision fallback
    # ------------------------------------------------------------------

    def _gemini_vision_extract(
        self, pdf_path: Path, page_number: int
    ) -> list[ExtractedTable] | None:
        """Use Gemini Vision to extract tables from a rendered page image."""
        if not settings.gemini_api_key:
            logger.warning("No GEMINI_API_KEY configured; skipping vision fallback")
            return None

        try:
            import google.generativeai as genai
        except ImportError:
            logger.warning("google-generativeai not installed; skipping vision fallback")
            return None

        genai.configure(api_key=settings.gemini_api_key)

        # Render page to image
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_number]
            pix = page.get_pixmap(dpi=_VISION_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        prompt = (
            "Extract ALL tables from this research paper page as JSON.\n"
            "Return a JSON array where each element has:\n"
            '  - "headers": list of column header strings\n'
            '  - "rows": list of lists of cell value strings\n'
            "If no tables exist, return an empty array [].\n"
            "Return ONLY valid JSON, no markdown fences."
        )

        model = genai.GenerativeModel(settings.gemini_model)
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
        except Exception:
            logger.error("Gemini vision extraction failed", exc_info=True)
            return None

        if not isinstance(data, list):
            data = [data]

        results: list[ExtractedTable] = []
        for idx, tbl in enumerate(data):
            if not isinstance(tbl, dict):
                continue
            results.append(
                ExtractedTable(
                    page_number=page_number,
                    table_index=idx,
                    headers=tbl.get("headers", []),
                    rows=tbl.get("rows", []),
                    extraction_method="gemini_vision",
                    confidence=0.75,
                )
            )

        return results if results else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_raw_table(raw_table: list[list[str | None]]) -> tuple[list[str], list[list[str]]]:
        """Split a raw pdfplumber table into headers and data rows."""
        if not raw_table:
            return [], []

        headers = [cell.strip() if cell else "" for cell in raw_table[0]]
        rows = [
            [cell.strip() if cell else "" for cell in row]
            for row in raw_table[1:]
        ]
        return headers, rows

    @staticmethod
    def _score_table(raw_table: list[list[str | None]]) -> float:
        """Score a table's quality in [0, 1]."""
        if not raw_table:
            return 0.0
        total = sum(len(row) for row in raw_table)
        if total == 0:
            return 0.0
        filled = sum(1 for row in raw_table for cell in row if cell and cell.strip())
        return round(filled / total, 4)

    @staticmethod
    def _find_caption(page_text: str, table_index: int) -> str | None:
        """Heuristic search for a table caption in the page text.

        Looks for lines starting with "Table N" (case-insensitive).
        """
        import re

        pattern = rf"(?i)table\s+{table_index + 1}[\s:.]\s*(.+)"
        match = re.search(pattern, page_text)
        if match:
            return f"Table {table_index + 1}: {match.group(1).strip()}"

        # Fallback: any line starting with "Table"
        for line in page_text.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("table") and len(stripped) > 6:
                return stripped
        return None

    @staticmethod
    def _get_page_texts(pdf_path: Path) -> dict[int, str]:
        """Quick per-page text extraction for caption searching."""
        texts: dict[int, str] = {}
        doc = fitz.open(str(pdf_path))
        try:
            for i in range(len(doc)):
                texts[i] = doc[i].get_text("text") or ""
        finally:
            doc.close()
        return texts
