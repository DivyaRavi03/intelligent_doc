"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.schemas import (
    DetectedSection,
    DocumentMetadataSchema,
    EnrichedChunk,
    ExtractionMethod,
    ExtractedTable,
    PageResult,
    PaperStructure,
    SectionType,
)


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


# ---------------------------------------------------------------------------
# Phase 2 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_paper(sample_pages: list[PageResult]) -> PaperStructure:
    """A fully-parsed paper structure for chunking tests."""
    return PaperStructure(
        paper_id="test-paper-001",
        sections=[
            DetectedSection(
                section_type=SectionType.TITLE,
                title=None,
                text="Deep Learning for Document Understanding\nAlice Smith, Bob Jones",
                page_start=0,
                page_end=0,
                order_index=0,
            ),
            DetectedSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                text=(
                    "This paper presents a novel approach to document understanding "
                    "using deep neural networks. We demonstrate state-of-the-art results "
                    "on multiple benchmarks including DocBank and PubLayNet. Our method "
                    "combines visual features with textual representations to achieve "
                    "robust performance across diverse document layouts."
                ),
                page_start=0,
                page_end=0,
                order_index=1,
            ),
            DetectedSection(
                section_type=SectionType.INTRODUCTION,
                title="1. Introduction",
                text=(
                    "Document understanding is a critical task in information extraction. "
                    "Prior work has focused on rule-based systems, but recent advances in "
                    "deep learning have opened new possibilities. " * 20  # ~1200 chars
                ),
                page_start=1,
                page_end=1,
                order_index=2,
            ),
            DetectedSection(
                section_type=SectionType.METHODOLOGY,
                title="2. Methodology",
                text=(
                    "We propose a multi-modal transformer architecture that jointly models "
                    "text content and visual layout features. The model uses a BERT-based "
                    "encoder for text and a ResNet backbone for visual features. "
                    "The loss function is $$L = L_{cls} + \\lambda L_{layout}$$ "
                    "where $\\lambda$ controls the layout weight."
                ),
                page_start=1,
                page_end=2,
                order_index=3,
            ),
            DetectedSection(
                section_type=SectionType.RESULTS,
                title="4. Results",
                text="Our model achieves 94.2% F1 on DocBank and 91.8% mAP on PubLayNet.",
                page_start=2,
                page_end=2,
                order_index=4,
            ),
            DetectedSection(
                section_type=SectionType.CONCLUSION,
                title="5. Conclusion",
                text="We presented a deep learning approach for document understanding.",
                page_start=2,
                page_end=2,
                order_index=5,
            ),
            DetectedSection(
                section_type=SectionType.REFERENCES,
                title="References",
                text=(
                    "[1] Smith et al. Deep Learning for NLP. ICML 2023.\n"
                    "[2] Jones and Lee. Transformer Architectures. NeurIPS 2023.\n"
                    "[3] Wang et al. Layout Analysis with CNNs. CVPR 2022.\n"
                    "[4] Chen et al. Document AI Survey. ACL 2023.\n"
                    "[5] Li et al. Multi-modal Learning. AAAI 2023.\n"
                    "[6] Zhang et al. Visual Feature Extraction. ECCV 2022.\n"
                    "[7] Brown et al. Language Models. NeurIPS 2020.\n"
                    "[8] Kim et al. Table Detection. ICDAR 2023.\n"
                    "[9] Park et al. OCR Systems. CVPR 2023.\n"
                    "[10] Liu et al. Pre-training Methods. ICLR 2023.\n"
                    "[11] Yang et al. Layout Analysis. AAAI 2022.\n"
                    "[12] Wu et al. Attention Mechanisms. ACL 2022."
                ),
                page_start=3,
                page_end=3,
                order_index=6,
            ),
        ],
        tables=[
            ExtractedTable(
                page_number=2,
                table_index=0,
                headers=["Method", "F1", "mAP"],
                rows=[
                    ["Ours", "94.2", "91.8"],
                    ["LayoutLM", "89.1", "86.3"],
                    ["BERT-base", "82.5", "79.1"],
                ],
                caption="Table 1: Performance comparison on benchmarks.",
                extraction_method="pdfplumber",
                confidence=0.95,
            ),
        ],
        references=[
            "[1] Smith et al. Deep Learning for NLP. ICML 2023.",
            "[2] Jones and Lee. Transformer Architectures. NeurIPS 2023.",
            "[3] Wang et al. Layout Analysis with CNNs. CVPR 2022.",
            "[4] Chen et al. Document AI Survey. ACL 2023.",
            "[5] Li et al. Multi-modal Learning. AAAI 2023.",
            "[6] Zhang et al. Visual Feature Extraction. ECCV 2022.",
            "[7] Brown et al. Language Models. NeurIPS 2020.",
            "[8] Kim et al. Table Detection. ICDAR 2023.",
            "[9] Park et al. OCR Systems. CVPR 2023.",
            "[10] Liu et al. Pre-training Methods. ICLR 2023.",
            "[11] Yang et al. Layout Analysis. AAAI 2022.",
            "[12] Wu et al. Attention Mechanisms. ACL 2022.",
        ],
        metadata=DocumentMetadataSchema(
            title="Deep Learning for Document Understanding",
            authors=["Alice Smith", "Bob Jones"],
            abstract="This paper presents a novel approach...",
            doi="10.1234/example.2024.001",
            keywords=["deep learning", "document AI"],
            confidence=0.85,
        ),
    )


@pytest.fixture()
def sample_chunks(sample_paper: PaperStructure) -> list[EnrichedChunk]:
    """Pre-built enriched chunks for embedding and vector store tests."""
    from src.chunking.chunker import SectionAwareChunker

    chunker = SectionAwareChunker(target_tokens=512, overlap_tokens=50)
    return chunker.chunk(sample_paper)


# ---------------------------------------------------------------------------
# Phase 3 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_search_results():
    """Pre-built SearchResult objects for retrieval tests."""
    from src.retrieval.vector_store import SearchResult

    topics = ["deep learning", "transformers", "attention", "NLP", "embeddings"]
    return [
        SearchResult(
            chunk_id=f"chunk-{i}",
            text=f"Sample text for chunk {i} about {topics[i % 5]}",
            score=1.0 - (i * 0.1),
            paper_id="paper-001",
            section_type="introduction",
            section_title="1. Introduction",
            page_numbers=[1],
            metadata={},
        )
        for i in range(5)
    ]


@pytest.fixture()
def sample_ranked_results():
    """Pre-built RankedResult objects for reranker and processor tests."""
    from src.retrieval.hybrid_retriever import RankedResult

    sections = ["introduction", "methodology", "results", "conclusion", "abstract"]
    return [
        RankedResult(
            chunk_id=f"chunk-{i}",
            text=f"Sample passage {i} about topic",
            paper_id="paper-001",
            section_type=sections[i % 5],
            rrf_score=1.0 / (60 + i + 1),
        )
        for i in range(5)
    ]
