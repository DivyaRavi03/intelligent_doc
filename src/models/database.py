"""SQLAlchemy ORM models for the Intelligent Document Processing platform."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Document(Base):
    """A research paper uploaded to the platform."""

    __tablename__ = "documents"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: str = Column(String(512), nullable=False)
    file_path: str = Column(String(1024), nullable=False)
    file_hash: str = Column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes: int = Column(Integer, nullable=False)
    num_pages: int = Column(Integer, nullable=True)
    status: str = Column(
        Enum("pending", "processing", "completed", "failed", name="document_status"),
        nullable=False,
        default="pending",
    )
    error_message: str | None = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: datetime = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    metadata_record = relationship("DocumentMetadata", back_populates="document", uselist=False)
    pages = relationship("Page", back_populates="document", order_by="Page.page_number")
    tables = relationship("TableRecord", back_populates="document")
    sections = relationship("Section", back_populates="document", order_by="Section.order_index")


class DocumentMetadata(Base):
    """Extracted metadata for a document (title, authors, etc.)."""

    __tablename__ = "document_metadata"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: uuid.UUID = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    title: str | None = Column(Text, nullable=True)
    authors: list | None = Column(JSONB, nullable=True)  # ["Author A", "Author B"]
    abstract: str | None = Column(Text, nullable=True)
    doi: str | None = Column(String(256), nullable=True)
    journal: str | None = Column(String(512), nullable=True)
    publication_date: str | None = Column(String(64), nullable=True)
    keywords: list | None = Column(JSONB, nullable=True)
    references_count: int | None = Column(Integer, nullable=True)
    extra: dict | None = Column(JSONB, nullable=True)
    confidence: float = Column(Float, nullable=False, default=0.0)

    document = relationship("Document", back_populates="metadata_record")

    __table_args__ = (
        Index("ix_document_metadata_document_id", "document_id", unique=True),
    )


class Page(Base):
    """Per-page text extraction result."""

    __tablename__ = "pages"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: uuid.UUID = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: int = Column(Integer, nullable=False)
    text: str = Column(Text, nullable=False, default="")
    extraction_method: str = Column(
        Enum("native", "ocr", name="extraction_method"), nullable=False, default="native"
    )
    confidence: float = Column(Float, nullable=False, default=1.0)
    char_count: int = Column(Integer, nullable=False, default=0)

    document = relationship("Document", back_populates="pages")

    __table_args__ = (
        Index("ix_pages_document_page", "document_id", "page_number", unique=True),
    )


class TableRecord(Base):
    """An extracted table from a document page."""

    __tablename__ = "tables"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: uuid.UUID = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: int = Column(Integer, nullable=False)
    table_index: int = Column(Integer, nullable=False)
    caption: str | None = Column(Text, nullable=True)
    headers: list | None = Column(JSONB, nullable=True)
    data: list | None = Column(JSONB, nullable=True)  # list of row dicts
    extraction_method: str = Column(String(64), nullable=False, default="pdfplumber")
    confidence: float = Column(Float, nullable=False, default=1.0)

    document = relationship("Document", back_populates="tables")

    __table_args__ = (
        Index("ix_tables_document_page", "document_id", "page_number"),
    )


class Section(Base):
    """A structural section detected in a document."""

    __tablename__ = "sections"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: uuid.UUID = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    section_type: str = Column(String(64), nullable=False)
    title: str | None = Column(Text, nullable=True)
    text: str = Column(Text, nullable=False, default="")
    page_start: int = Column(Integer, nullable=False)
    page_end: int = Column(Integer, nullable=True)
    order_index: int = Column(Integer, nullable=False, default=0)

    document = relationship("Document", back_populates="sections")

    __table_args__ = (
        Index("ix_sections_document_order", "document_id", "order_index"),
    )
