"""Query, search, comparison, and summarization endpoints.

All LLM-backed services are lazily initialized to avoid import-time
failures when API keys are not configured.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import verify_api_key
from src.api.rate_limiter import QUERY_LIMIT, limiter
from src.api.stores import (
    InMemoryDocumentStore,
    get_document_store,
    update_metrics,
)
from src.models.schemas import (
    CompareRequest,
    CompareResponse,
    DocumentStatus,
    QAResponse,
    QueryRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SummaryLevel,
    SummaryResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


# ---------------------------------------------------------------------------
# Lazy service initialization
# ---------------------------------------------------------------------------

_qa_engine = None
_retriever = None
_summarizer = None
_client = None


def _get_client():  # type: ignore[no-untyped-def]
    global _client
    if _client is None:
        from src.llm.gemini_client import GeminiClient
        _client = GeminiClient()
    return _client


def _get_retriever():  # type: ignore[no-untyped-def]
    global _retriever
    if _retriever is None:
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.embedding_service import EmbeddingService
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore()
        es = EmbeddingService()
        bm = BM25Index()
        _retriever = HybridRetriever(vs, es, bm)
    return _retriever


def _get_qa_engine():  # type: ignore[no-untyped-def]
    global _qa_engine
    if _qa_engine is None:
        from src.llm.qa_engine import QAEngine
        _qa_engine = QAEngine(client=_get_client(), retriever=_get_retriever())
    return _qa_engine


def _get_summarizer():  # type: ignore[no-untyped-def]
    global _summarizer
    if _summarizer is None:
        from src.llm.summarizer import PaperSummarizer
        _summarizer = PaperSummarizer(client=_get_client())
    return _summarizer


_cross_paper = None


def _get_cross_paper_analyzer():  # type: ignore[no-untyped-def]
    global _cross_paper
    if _cross_paper is None:
        from src.llm.cross_paper import CrossPaperAnalyzer
        _cross_paper = CrossPaperAnalyzer(client=_get_client(), store=get_document_store())
    return _cross_paper


# ------------------------------------------------------------------
# POST /query
# ------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QAResponse,
    summary="Ask a question about papers",
    description=(
        "Submit a natural-language question. The QA engine retrieves "
        "relevant passages, generates a cited answer, and verifies each "
        "claim for faithfulness."
    ),
)
@limiter.limit(QUERY_LIMIT)
async def query_documents(
    request: Request,
    body: QueryRequest,
    api_key: str = Depends(verify_api_key),
) -> QAResponse:
    """Answer a question using citation-tracked QA."""
    logger.info("Query: %s", body.query)
    try:
        qa_engine = _get_qa_engine()
        response = qa_engine.answer(
            query=body.query,
            paper_ids=body.paper_ids,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("QA engine error")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    update_metrics("total_queries", 1)
    return response


# ------------------------------------------------------------------
# POST /search
# ------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search across papers",
    description=(
        "Hybrid dense + sparse search with reciprocal rank fusion. "
        "Returns ranked passages with scores."
    ),
)
@limiter.limit(QUERY_LIMIT)
async def search_documents(
    request: Request,
    body: SearchRequest,
    api_key: str = Depends(verify_api_key),
) -> SearchResponse:
    """Hybrid search across all indexed papers."""
    logger.info("Search: %s (top_k=%d)", body.query, body.top_k)
    try:
        retriever = _get_retriever()
        results = retriever.retrieve(
            query=body.query,
            top_k=body.top_k,
            alpha=body.alpha,
            paper_id=body.paper_id,
            section_type=body.section_type,
        )
    except Exception as exc:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.final_score if r.final_score else r.rrf_score,
            paper_id=r.paper_id,
            section_type=r.section_type,
            page_numbers=getattr(r, "page_numbers", []),
        )
        for r in results
    ]

    return SearchResponse(query=body.query, results=items, total_results=len(items))


# ------------------------------------------------------------------
# POST /compare
# ------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare findings across papers",
    description="Generate a comparative analysis of multiple papers on a given aspect.",
)
@limiter.limit(QUERY_LIMIT)
async def compare_papers(
    request: Request,
    body: CompareRequest,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> CompareResponse:
    """Compare multiple papers on a specific aspect."""
    logger.info("Compare: %s on aspect=%s", body.paper_ids, body.aspect)

    try:
        analyzer = _get_cross_paper_analyzer()
        result = analyzer.compare_papers(body.paper_ids, body.aspect)

        return CompareResponse(
            aspect=result.aspect,
            papers={pid: "included" for pid in result.paper_ids},
            comparison_text=result.synthesis,
            key_differences=result.contradictions[:5],
        )
    except Exception as exc:
        logger.exception("Comparison error")
        raise HTTPException(
            status_code=500, detail=f"Comparison failed: {exc}"
        ) from exc


# ------------------------------------------------------------------
# GET /documents/{doc_id}/summary/{level}
# ------------------------------------------------------------------


@router.get(
    "/documents/{doc_id}/summary/{level}",
    response_model=SummaryResult,
    summary="Get paper summary",
    description="Generate a summary at the specified detail level (one_line, abstract, detailed).",
)
async def get_summary(
    doc_id: uuid.UUID,
    level: SummaryLevel,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> SummaryResult:
    """Summarize a paper at the requested level of detail."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not completed (status: {doc.status.value})",
        )

    paper_text = "\n\n".join(s.text for s in doc.sections)
    if not paper_text.strip():
        raise HTTPException(status_code=400, detail="Document has no text content")

    try:
        summarizer = _get_summarizer()
        result = summarizer.summarize(
            paper_id=str(doc_id),
            text=paper_text,
            sections=doc.sections if doc.sections else None,
            level=level.value,
        )
    except Exception as exc:
        logger.exception("Summarization error")
        raise HTTPException(
            status_code=500, detail=f"Summarization failed: {exc}"
        ) from exc

    return result
