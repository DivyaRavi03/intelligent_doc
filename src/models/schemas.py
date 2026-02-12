"""Pydantic schemas for request/response validation and internal data transfer."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionMethod(str, Enum):
    NATIVE = "native"
    OCR = "ocr"


class SectionType(str, Enum):
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


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = "0.1.0"
