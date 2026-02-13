"""Pydantic schemas for request/response validation and internal data transfer."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionMethod(StrEnum):
    NATIVE = "native"
    OCR = "ocr"


class SectionType(StrEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Page-level schemas
# ---------------------------------------------------------------------------

class PageResult(BaseModel):
    """Result of text extraction for a single page."""

    page_number: int
    text: str
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    char_count: int = Field(ge=0)


class PDFExtractionResult(BaseModel):
    """Full result from the PDF parser."""

    pages: list[PageResult]
    total_pages: int
    avg_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

class ExtractedTable(BaseModel):
    """A single table extracted from a page."""

    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    extraction_method: str = "pdfplumber"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class TableExtractionResult(BaseModel):
    """All tables extracted from a document."""

    tables: list[ExtractedTable]
    total_tables: int


# ---------------------------------------------------------------------------
# Layout / section schemas
# ---------------------------------------------------------------------------

class DetectedSection(BaseModel):
    """A structural section identified during layout analysis."""

    section_type: SectionType
    title: str | None = None
    text: str
    page_start: int
    page_end: int | None = None
    order_index: int = 0


class LayoutAnalysisResult(BaseModel):
    """Full layout analysis output."""

    sections: list[DetectedSection]
    references: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metadata schemas
# ---------------------------------------------------------------------------

class DocumentMetadataSchema(BaseModel):
    """Extracted metadata for a research paper."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    journal: str | None = None
    publication_date: str | None = None
    keywords: list[str] = Field(default_factory=list)
    references_count: int | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class DocumentUploadResponse(BaseModel):
    """Response after a document is uploaded."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    num_pages: int | None = None
    task_id: str | None = None
    created_at: datetime


class DocumentDetailResponse(BaseModel):
    """Full document detail including extracted data."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    num_pages: int | None = None
    file_size_bytes: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: DocumentMetadataSchema | None = None
    sections: list[DetectedSection] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking schemas
# ---------------------------------------------------------------------------

class EnrichedChunk(BaseModel):
    """A text chunk enriched with structural and provenance metadata."""

    chunk_id: str
    text: str
    token_count: int = Field(ge=0)
    section_type: SectionType
    section_title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    paper_id: str
    paper_title: str | None = None
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    metadata: dict = Field(default_factory=dict)


class PaperStructure(BaseModel):
    """Aggregated output from Phase 1 that feeds into the chunker.

    Bundles sections, tables, references, and metadata into a single
    object representing a fully-parsed paper.
    """

    paper_id: str
    sections: list[DetectedSection] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    metadata: DocumentMetadataSchema = Field(default_factory=DocumentMetadataSchema)


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Phase 4 — LLM response schemas
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Response from GeminiClient.generate()."""

    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    model: str
    cached: bool = False
    cost_usd: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Phase 4 — Extraction schemas
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single key finding extracted from a paper."""

    claim: str
    supporting_quote: str
    confidence: float = Field(ge=0.0, le=1.0)


class MethodologyExtraction(BaseModel):
    """Structured methodology from a paper."""

    approach: str
    datasets: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    eval_metrics: list[str] = Field(default_factory=list)


class ResultExtraction(BaseModel):
    """A single quantitative result."""

    metric_name: str
    value: str
    baseline: str | None = None
    improvement: str | None = None
    table_reference: str | None = None


class PaperExtraction(BaseModel):
    """Complete extraction from a single paper."""

    paper_id: str
    key_findings: list[Finding] = Field(default_factory=list)
    methodology: MethodologyExtraction | None = None
    results: list[ResultExtraction] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    needs_review: bool = False


# ---------------------------------------------------------------------------
# Phase 4 — QA schemas
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A source citation in an answer."""

    source_index: int = Field(ge=1)
    chunk_id: str
    text_snippet: str
    paper_id: str
    section_type: str
    page_numbers: list[int] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    """Verification status of a single claim in an answer."""

    claim: str
    cited_source_index: int | None = None
    status: str = Field(description="SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED")
    explanation: str = ""


class QAResponse(BaseModel):
    """Full QA response with answer, citations, and verification."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    faithfulness_score: float = Field(ge=0.0, le=1.0, default=0.0)
    flagged_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 4 — Summarization schemas
# ---------------------------------------------------------------------------

class SummaryLevel(StrEnum):
    ONE_LINE = "one_line"
    ABSTRACT = "abstract"
    DETAILED = "detailed"


class SummaryResult(BaseModel):
    """Result of paper summarization."""

    paper_id: str
    level: SummaryLevel
    summary: str
    word_count: int = Field(ge=0)
    sections_used: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 4 — Feedback schemas
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """Request body for POST /api/v1/feedback."""

    paper_id: str
    field_name: str
    original_value: str
    corrected_value: str
    user_comment: str = ""


class FeedbackResponse(BaseModel):
    """Response after feedback is recorded."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    paper_id: str
    field_name: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Phase 5 — Document list schemas
# ---------------------------------------------------------------------------

class DocumentListItem(BaseModel):
    """Minimal document info for list view."""

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    num_pages: int | None = None
    created_at: datetime
    metadata_title: str | None = None


class DocumentListResponse(BaseModel):
    """Paginated document list."""

    documents: list[DocumentListItem]
    total: int
    offset: int = 0
    limit: int = 20


# ---------------------------------------------------------------------------
# Phase 5 — Query / search schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /api/v1/query."""

    query: str = Field(min_length=1, max_length=2000, description="Question about papers")
    paper_ids: list[str] | None = Field(
        default=None, description="Restrict to specific paper IDs"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages")


class SearchRequest(BaseModel):
    """Request body for POST /api/v1/search."""

    query: str = Field(min_length=1, max_length=2000, description="Search query")
    paper_id: str | None = Field(default=None, description="Restrict to one paper")
    section_type: str | None = Field(default=None, description="Filter by section type")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Dense vs sparse weight (1.0 = all dense)"
    )


class SearchResultItem(BaseModel):
    """A single search result."""

    chunk_id: str
    text: str
    score: float
    paper_id: str
    section_type: str
    page_numbers: list[int] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Search results response."""

    query: str
    results: list[SearchResultItem]
    total_results: int


class CompareRequest(BaseModel):
    """Request body for POST /api/v1/compare."""

    paper_ids: list[str] = Field(min_length=2, max_length=5, description="Papers to compare")
    aspect: str = Field(default="methodology", description="Comparison aspect")


class CompareResponse(BaseModel):
    """Paper comparison response."""

    aspect: str
    papers: dict[str, str] = Field(description="paper_id → aspect summary")
    comparison_text: str
    key_differences: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 8 — Cross-paper comparison schemas
# ---------------------------------------------------------------------------

class ComparisonTableRow(BaseModel):
    """A single row in a cross-paper comparison table."""

    aspect: str
    papers: dict[str, str] = Field(description="paper_id → value for that aspect")


class CrossPaperComparison(BaseModel):
    """Structured cross-paper comparison result."""

    paper_ids: list[str]
    aspect: str
    comparison_table: list[ComparisonTableRow] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    synthesis: str = ""
    model_used: str = ""


# ---------------------------------------------------------------------------
# Phase 5 — Extraction management schemas
# ---------------------------------------------------------------------------

class ReExtractRequest(BaseModel):
    """Request body for POST /api/v1/documents/{id}/re-extract."""

    extraction_types: list[str] = Field(
        default_factory=lambda: ["findings", "methodology", "results"],
        description="Which extractions to re-run",
    )
    force: bool = Field(default=False, description="Force re-extraction even if cached")


class ExtractionStatsResponse(BaseModel):
    """Extraction feedback statistics."""

    total_feedback: int
    by_field: dict[str, int] = Field(default_factory=dict)
    flagged_papers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 5 — Admin schemas
# ---------------------------------------------------------------------------

class HealthCheckResponse(BaseModel):
    """Detailed health check response for all services."""

    status: str = "ok"
    version: str = "0.1.0"
    database: str = "ok"
    redis: str = "ok"
    chromadb: str = "ok"
    timestamp: datetime


class MetricsResponse(BaseModel):
    """System usage metrics."""

    total_documents: int = 0
    total_queries: int = 0
    total_llm_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_processing_time_sec: float | None = None
    error_rate: float = 0.0


class CostBreakdown(BaseModel):
    """LLM cost breakdown by operation type."""

    gemini_embedding_cost: float = 0.0
    gemini_generation_cost: float = 0.0
    total_cost: float = 0.0
    documents_processed: int = 0
    queries_answered: int = 0
    avg_cost_per_document: float = 0.0


class EvaluationResult(BaseModel):
    """Latest evaluation results from test set."""

    timestamp: datetime
    test_set_name: str = ""
    accuracy: float = 0.0
    faithfulness_score: float = 0.0
    avg_latency_ms: float = 0.0
    pass_rate: float = 0.0


# ---------------------------------------------------------------------------
# Phase 5 — WebSocket / task schemas
# ---------------------------------------------------------------------------

class ProcessingUpdate(BaseModel):
    """WebSocket message for real-time processing updates."""

    task_id: str
    status: str = Field(description="queued | processing | completed | failed")
    stage: str | None = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    step: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    result: dict | None = None


class TaskStatusResponse(BaseModel):
    """Response for GET /api/v1/documents/{id}/status."""

    document_id: uuid.UUID
    status: DocumentStatus
    task_id: str | None = None
    progress: float = 0.0
    current_stage: str | None = None
    error_message: str | None = None
