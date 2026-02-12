"""Embedding service backed by Google's text-embedding-004 model.

The :class:`EmbeddingService` handles batching, rate-limit delays, and
exponential-backoff retry for embedding both document chunks and search
queries.  It distinguishes between indexing (``task_type='retrieval_document'``)
and search (``task_type='retrieval_query'``) following Google's embedding
best-practices.
"""

from __future__ import annotations

import logging
import time

from src.config import settings
from src.models.schemas import EnrichedChunk

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100
_DEFAULT_DELAY = 0.5  # seconds between batches
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0  # seconds


class EmbeddingService:
    """Embed text chunks and queries using Google's text-embedding-004.

    Args:
        model_name: Embedding model identifier.
        batch_size: Maximum texts per API call.
        delay: Seconds to sleep between batches.
        max_retries: Maximum retries per batch on transient failure.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        delay: float = _DEFAULT_DELAY,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.delay = delay
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self, chunks: list[EnrichedChunk]
    ) -> list[list[float]]:
        """Embed document chunks for indexing.

        Args:
            chunks: Enriched chunks to embed.

        Returns:
            List of embedding vectors, one per chunk, in the same order.

        Raises:
            RuntimeError: If the Gemini API key is not configured.
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("No chunks to embed")
        texts = [c.text for c in chunks]
        return self._batch_embed(texts, task_type="retrieval_document")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Args:
            query: The search query text.

        Returns:
            A single embedding vector.

        Raises:
            RuntimeError: If the Gemini API key is not configured.
            ValueError: If *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        result = self._batch_embed([query], task_type="retrieval_query")
        return result[0]

    # ------------------------------------------------------------------
    # Internal batching with retry
    # ------------------------------------------------------------------

    def _batch_embed(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """Embed texts in batches with rate-limit delays and retry.

        Args:
            texts: Raw strings to embed.
            task_type: ``'retrieval_document'`` for indexing or
                ``'retrieval_query'`` for search.

        Returns:
            List of embedding vectors, one per input text.
        """
        self._ensure_api_key()

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            embeddings = self._embed_with_retry(genai, batch, task_type)
            all_embeddings.extend(embeddings)

            # Rate-limit delay between batches (skip after last batch)
            if batch_start + self.batch_size < len(texts):
                time.sleep(self.delay)

        return all_embeddings

    def _embed_with_retry(
        self,
        genai,  # noqa: ANN001  — google.generativeai module
        batch: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """Call the embedding API with exponential-backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type=task_type,
                )
                # result["embedding"] is a list of vectors when input is a list
                embedding = result["embedding"]
                if isinstance(embedding[0], float):
                    # Single text was passed — wrap in a list
                    return [embedding]
                return embedding
            except Exception as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Embedding attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_api_key() -> None:
        """Raise immediately if no API key is configured."""
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )
