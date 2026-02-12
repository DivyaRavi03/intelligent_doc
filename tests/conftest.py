"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.schemas import ExtractionMethod, PageResult


@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF for testing.

    Uses PyMuPDF to generate a two-page PDF with known text content.
    """
    import fitz

    doc = fitz.open()

    # Page 0 — title / abstract
    page0 = doc.new_page(width=612, height=792)
    page0.insert_text(
        (72, 80),
        "Deep Learning for Document Understanding",
        fontsize=16,
    )
    page0.insert_text(
        (72, 110),
        "Alice Smith, Bob Jones",
        fontsize=11,
    )
    page0.insert_text(
        (72, 140),
        "Abstract",
        fontsize=13,
    )
    page0.insert_text(
        (72, 160),
        textwrap.fill(
            "This paper presents a novel approach to document understanding "
            "using deep neural networks. We demonstrate state-of-the-art results "
            "on multiple benchmarks including DocBank and PubLayNet.",
            width=80,
        ),
        fontsize=10,
    )
    page0.insert_text((72, 250), "Keywords: deep learning, document AI, layout analysis", fontsize=9)
    page0.insert_text((72, 280), "DOI: 10.1234/example.2024.001", fontsize=9)

    # Page 1 — introduction + references
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((72, 80), "1. Introduction", fontsize=13)
    page1.insert_text(
        (72, 110),
        textwrap.fill(
            "Document understanding is a critical task in information extraction. "
            "Prior work has focused on rule-based systems, but recent advances in "
            "deep learning have opened new possibilities.",
            width=80,
        ),
        fontsize=10,
    )
    page1.insert_text((72, 250), "References", fontsize=13)
    page1.insert_text(
        (72, 280),
        "[1] Smith et al. Deep Learning for NLP. ICML 2023.",
        fontsize=9,
    )
    page1.insert_text(
        (72, 300),
        "[2] Jones and Lee. Transformer Architectures. NeurIPS 2023.",
        fontsize=9,
    )

    pdf_path = tmp_path / "test_paper.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def sample_pages() -> list[PageResult]:
    """Pre-built page results for testing layout and metadata extractors."""
    return [
        PageResult(
            page_number=0,
            text=(
                "Deep Learning for Document Understanding\n"
                "Alice Smith, Bob Jones\n"
                "\n"
                "Abstract\n"
                "This paper presents a novel approach to document understanding "
                "using deep neural networks. We demonstrate state-of-the-art results "
                "on multiple benchmarks including DocBank and PubLayNet.\n"
                "\n"
                "Keywords: deep learning, document AI, layout analysis\n"
                "DOI: 10.1234/example.2024.001\n"
            ),
            extraction_method=ExtractionMethod.NATIVE,
            confidence=0.95,
            char_count=350,
        ),
        PageResult(
            page_number=1,
            text=(
                "1. Introduction\n"
                "Document understanding is a critical task in information extraction. "
                "Prior work has focused on rule-based systems, but recent advances in "
                "deep learning have opened new possibilities.\n"
                "\n"
                "2. Methodology\n"
                "We propose a multi-modal transformer architecture that jointly models "
                "text content and visual layout features.\n"
                "\n"
                "3. Experiments\n"
                "We evaluate our approach on three benchmark datasets.\n"
            ),
            extraction_method=ExtractionMethod.NATIVE,
            confidence=0.92,
            char_count=400,
        ),
        PageResult(
            page_number=2,
            text=(
                "4. Results\n"
                "Our model achieves 94.2% F1 on DocBank and 91.8% mAP on PubLayNet.\n"
                "\n"
                "5. Conclusion\n"
                "We presented a deep learning approach for document understanding.\n"
                "\n"
                "References\n"
                "[1] Smith et al. Deep Learning for NLP. ICML 2023.\n"
                "[2] Jones and Lee. Transformer Architectures. NeurIPS 2023.\n"
                "[3] Wang et al. Layout Analysis with CNNs. CVPR 2022.\n"
            ),
            extraction_method=ExtractionMethod.NATIVE,
            confidence=0.90,
            char_count=350,
        ),
    ]


@pytest.fixture()
def mock_fitz_page() -> MagicMock:
    """A mocked fitz.Page with realistic return values."""
    page = MagicMock()
    page.get_text.return_value = "Sample text content for testing purposes.\nSecond line here."
    page.rect = MagicMock()
    page.rect.width = 612
    page.rect.height = 792
    page.get_text.side_effect = lambda fmt="text", **kw: (
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"spans": [{"text": "Sample text content for testing purposes."}]},
                        {"spans": [{"text": "Second line here."}]},
                    ],
                }
            ]
        }
        if fmt == "dict"
        else "Sample text content for testing purposes.\nSecond line here."
    )
    return page
