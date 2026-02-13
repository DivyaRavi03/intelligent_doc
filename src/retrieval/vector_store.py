"""ChromaDB vector store for chunk storage and semantic search.

The :class:`VectorStore` provides persistent, metadata-filtered similarity
search over embedded document chunks.  It uses **cosine similarity** as the
distance metric and supports filtering by ``paper_id`` and ``section_type``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import settings
from src.models.schemas import EnrichedChunk, SectionType

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "paper_chunks"


@dataclass
class SearchResult:
    """A single search hit from the vector store."""

    chunk_id: str
    text: str
    score: float
    paper_id: str
    section_type: str
    section_title: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Persistent ChromaDB vector store for enriched chunks.

    Args:
        persist_dir: Directory for ChromaDB on-disk storage.
            Defaults to the value from application settings.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name
        self._client: object = None
        self._collection: object = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_collection(self) -> object:
        """Return the ChromaDB collection, creating it on first access."""
        if self._collection is not None:
            return self._collection

        import chromadb

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d items)",
            self.collection_name,
            self._collection.count(),
        )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        chunks: list[EnrichedChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Store embedded chunks in the vector store.

        Args:
            chunks: Enriched chunks with metadata.
            embeddings: Embedding vectors, one per chunk.

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If *chunks* and *embeddings* have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return 0

        collection = self._get_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embedding_list: list[list[float]] = []

        for chunk, emb in zip(chunks, embeddings, strict=True):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._chunk_to_metadata(chunk))
            embedding_list.append(emb)

        # ChromaDB upsert — idempotent, safe to re-run
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embedding_list,
            metadatas=metadatas,
        )

        logger.info("Stored %d chunks for paper %s", len(chunks), chunks[0].paper_id)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        paper_id: str | None = None,
        section_type: SectionType | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: The query's embedding vector.
            n_results: Maximum number of results.
            paper_id: Optional filter — only search within this paper.
            section_type: Optional filter — only search this section type.

        Returns:
            Ranked list of :class:`SearchResult` objects.
        """
        collection = self._get_collection()

        where: dict | None = self._build_where(paper_id, section_type)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            raw = collection.query(**kwargs)
        except Exception:
            logger.error("ChromaDB query failed", exc_info=True)
            return []

        return self._parse_results(raw)

    def delete_paper(self, paper_id: str) -> int:
        """Remove all chunks belonging to a specific paper.

        Args:
            paper_id: The paper whose chunks should be deleted.

        Returns:
            Number of chunks deleted.
        """
        collection = self._get_collection()

        # Get IDs of matching chunks
        results = collection.get(
            where={"paper_id": paper_id},
            include=[],
        )
        ids = results.get("ids", [])

        if not ids:
            logger.info("No chunks found for paper %s", paper_id)
            return 0

        collection.delete(ids=ids)
        logger.info("Deleted %d chunks for paper %s", len(ids), paper_id)
        return len(ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_metadata(chunk: EnrichedChunk) -> dict:
        """Flatten an EnrichedChunk into a ChromaDB-compatible metadata dict.

        ChromaDB metadata values must be str, int, float, or bool.
        """
        return {
            "paper_id": chunk.paper_id,
            "paper_title": chunk.paper_title or "",
            "section_type": chunk.section_type.value,
            "section_title": chunk.section_title or "",
            "page_numbers": ",".join(str(p) for p in chunk.page_numbers),
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.token_count,
        }

    @staticmethod
    def _build_where(
        paper_id: str | None,
        section_type: SectionType | None,
    ) -> dict | None:
        """Build a ChromaDB ``where`` filter dict."""
        conditions: list[dict] = []
        if paper_id:
            conditions.append({"paper_id": paper_id})
        if section_type:
            conditions.append({"section_type": section_type.value})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _parse_results(raw: dict) -> list[SearchResult]:
        """Convert raw ChromaDB query output to :class:`SearchResult` list."""
        results: list[SearchResult] = []

        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            distance = dists_list[i] if i < len(dists_list) else 1.0
            # ChromaDB returns cosine distance; convert to similarity
            score = 1.0 - distance

            page_str = meta.get("page_numbers", "")
            page_numbers = (
                [int(p) for p in page_str.split(",") if p.strip()]
                if page_str
                else []
            )

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=docs_list[i] if i < len(docs_list) else "",
                    score=round(score, 4),
                    paper_id=meta.get("paper_id", ""),
                    section_type=meta.get("section_type", "unknown"),
                    section_title=meta.get("section_title") or None,
                    page_numbers=page_numbers,
                    metadata=meta,
                )
            )

        return results
