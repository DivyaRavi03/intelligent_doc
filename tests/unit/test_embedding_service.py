"""Unit tests for the embedding service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.schemas import EnrichedChunk, SectionType
from src.retrieval.embedding_service import EmbeddingService


def _make_chunk(text: str, idx: int = 0) -> EnrichedChunk:
    """Helper to create a minimal EnrichedChunk for testing."""
    return EnrichedChunk(
        chunk_id=f"test-{idx}",
        text=text,
        token_count=len(text) // 4,
        section_type=SectionType.INTRODUCTION,
        paper_id="paper-001",
        chunk_index=idx,
        total_chunks=1,
    )


class TestEmbeddingService:
    """Tests for :class:`EmbeddingService`."""

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_chunks_raises_without_api_key(self, mock_settings: MagicMock) -> None:
        """Should raise RuntimeError when no API key is configured."""
        mock_settings.gemini_api_key = ""
        mock_settings.embedding_model = "models/text-embedding-004"
        service = EmbeddingService()
        chunks = [_make_chunk("test text")]

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            service.embed_chunks(chunks)

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_query_raises_without_api_key(self, mock_settings: MagicMock) -> None:
        """Should raise RuntimeError when no API key is configured."""
        mock_settings.gemini_api_key = ""
        mock_settings.embedding_model = "models/text-embedding-004"
        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            service.embed_query("test query")

    def test_embed_chunks_raises_on_empty_list(self) -> None:
        """Should raise ValueError for empty chunks list."""
        service = EmbeddingService()
        with pytest.raises(ValueError, match="No chunks"):
            service.embed_chunks([])

    def test_embed_query_raises_on_empty_string(self) -> None:
        """Should raise ValueError for empty query."""
        service = EmbeddingService()
        with pytest.raises(ValueError, match="empty"):
            service.embed_query("")

    def test_embed_query_raises_on_whitespace(self) -> None:
        """Should raise ValueError for whitespace-only query."""
        service = EmbeddingService()
        with pytest.raises(ValueError, match="empty"):
            service.embed_query("   ")

    # ------------------------------------------------------------------
    # Successful embedding (mocked API)
    # ------------------------------------------------------------------

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_chunks_returns_vectors(self, mock_settings: MagicMock) -> None:
        """embed_chunks should return one vector per chunk."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("google.generativeai.configure"):
            with patch(
                "google.generativeai.embed_content",
                return_value={"embedding": fake_embeddings},
            ):
                service = EmbeddingService(batch_size=10, delay=0.0)
                chunks = [_make_chunk("text one", 0), _make_chunk("text two", 1)]
                result = service.embed_chunks(chunks)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_query_returns_single_vector(self, mock_settings: MagicMock) -> None:
        """embed_query should return a single vector."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        with patch("google.generativeai.configure"):
            with patch(
                "google.generativeai.embed_content",
                return_value={"embedding": [0.1, 0.2, 0.3]},
            ):
                service = EmbeddingService(delay=0.0)
                result = service.embed_query("what is deep learning?")

        assert result == [0.1, 0.2, 0.3]

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    @patch("src.retrieval.embedding_service.settings")
    def test_batch_splitting(self, mock_settings: MagicMock) -> None:
        """Large chunk lists should be split into batches."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        call_count = 0
        batch_sizes: list[int] = []

        def mock_embed(model, content, task_type):
            nonlocal call_count
            call_count += 1
            batch_sizes.append(len(content))
            return {"embedding": [[0.1] * 3 for _ in content]}

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.embed_content", side_effect=mock_embed):
                service = EmbeddingService(batch_size=3, delay=0.0)
                chunks = [_make_chunk(f"text {i}", i) for i in range(7)]
                result = service.embed_chunks(chunks)

        assert len(result) == 7
        assert call_count == 3  # 3 + 3 + 1
        assert batch_sizes == [3, 3, 1]

    # ------------------------------------------------------------------
    # Task types
    # ------------------------------------------------------------------

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_chunks_uses_document_task_type(self, mock_settings: MagicMock) -> None:
        """Indexing should use task_type='retrieval_document'."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        captured_task_type = None

        def mock_embed(model, content, task_type):
            nonlocal captured_task_type
            captured_task_type = task_type
            return {"embedding": [[0.1] * 3 for _ in content]}

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.embed_content", side_effect=mock_embed):
                service = EmbeddingService(delay=0.0)
                service.embed_chunks([_make_chunk("test")])

        assert captured_task_type == "retrieval_document"

    @patch("src.retrieval.embedding_service.settings")
    def test_embed_query_uses_query_task_type(self, mock_settings: MagicMock) -> None:
        """Search should use task_type='retrieval_query'."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        captured_task_type = None

        def mock_embed(model, content, task_type):
            nonlocal captured_task_type
            captured_task_type = task_type
            return {"embedding": [0.1, 0.2, 0.3]}

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.embed_content", side_effect=mock_embed):
                service = EmbeddingService(delay=0.0)
                service.embed_query("test query")

        assert captured_task_type == "retrieval_query"

    # ------------------------------------------------------------------
    # Retry behaviour
    # ------------------------------------------------------------------

    @patch("src.retrieval.embedding_service.settings")
    @patch("src.retrieval.embedding_service.time.sleep")
    def test_retry_on_transient_failure(
        self, mock_sleep: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Should retry with exponential backoff on API failure."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        attempt = 0

        def mock_embed(model, content, task_type):
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("transient failure")
            return {"embedding": [[0.1] * 3 for _ in content]}

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.embed_content", side_effect=mock_embed):
                service = EmbeddingService(max_retries=5, delay=0.0)
                result = service.embed_chunks([_make_chunk("test")])

        assert len(result) == 1
        assert attempt == 3  # failed twice, succeeded on third
        # Backoff sleeps: 1.0s, 2.0s
        assert mock_sleep.call_count >= 2

    @patch("src.retrieval.embedding_service.settings")
    @patch("src.retrieval.embedding_service.time.sleep")
    def test_raises_after_max_retries(
        self, mock_sleep: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Should raise RuntimeError after exhausting retries."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.embedding_model = "models/text-embedding-004"

        with patch("google.generativeai.configure"):
            with patch(
                "google.generativeai.embed_content",
                side_effect=ConnectionError("persistent failure"),
            ):
                service = EmbeddingService(max_retries=2, delay=0.0)
                with pytest.raises(RuntimeError, match="failed after 2 retries"):
                    service.embed_chunks([_make_chunk("test")])

    # ------------------------------------------------------------------
    # Constructor defaults
    # ------------------------------------------------------------------

    def test_default_model_from_settings(self) -> None:
        """Default model should come from settings."""
        service = EmbeddingService()
        assert "embedding" in service.model_name

    def test_custom_parameters(self) -> None:
        """Custom constructor args should be stored."""
        service = EmbeddingService(
            model_name="custom-model",
            batch_size=50,
            delay=1.0,
            max_retries=3,
        )
        assert service.model_name == "custom-model"
        assert service.batch_size == 50
        assert service.delay == 1.0
        assert service.max_retries == 3
