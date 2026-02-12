"""Gemini-based passage reranker.

The :class:`GeminiReranker` sends all candidate passages to Gemini in a
single call, asking it to rate each passage's relevance to the query on
a 1–10 scale.  The reranker score is combined with the original RRF score
for the final ranking.  This two-stage approach (retrieve → rerank)
mirrors production search at Google and Bing.
"""

from __future__ import annotations

import json
import logging
import re
import time

from src.config import settings
from src.retrieval.hybrid_retriever import RankedResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0
_RERANK_WEIGHT = 0.7  # Weight for reranker score in final combination
_RRF_WEIGHT = 0.3  # Weight for original RRF score in final combination
_MAX_PASSAGE_CHARS = 500  # Truncate passages to stay within context limits


class GeminiReranker:
    """Rerank retrieval candidates using Gemini relevance scoring.

    Args:
        model_name: Gemini model to use.  Defaults to ``settings.gemini_model``.
        max_retries: Maximum retries on transient API failures.
        rerank_weight: Weight of reranker score in the final combination.
        rrf_weight: Weight of the original RRF score in the final combination.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = _MAX_RETRIES,
        rerank_weight: float = _RERANK_WEIGHT,
        rrf_weight: float = _RRF_WEIGHT,
    ) -> None:
        self._model_name = model_name or settings.gemini_model
        self._max_retries = max_retries
        self._rerank_weight = rerank_weight
        self._rrf_weight = rrf_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        """Rerank candidates by relevance to the query.

        Sends all candidates to Gemini in a single call.  Falls back to
        original RRF ordering on any failure.

        Args:
            query: The search query.
            candidates: Fused results from the hybrid retriever.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of :class:`RankedResult` with updated scores.
        """
        if not candidates or not query:
            return candidates[:top_k]

        if not settings.gemini_api_key:
            logger.warning("No API key configured, skipping reranking")
            return candidates[:top_k]

        prompt = self._build_prompt(query, candidates)

        try:
            response_text = self._call_gemini(prompt)
        except RuntimeError:
            logger.warning("Gemini reranking failed, returning original order")
            return candidates[:top_k]

        scores = self._parse_scores(response_text, len(candidates))
        if not scores:
            logger.warning("Failed to parse reranker scores, returning original order")
            return candidates[:top_k]

        # Apply scores and compute final ranking
        reranked: list[RankedResult] = []
        for i, candidate in enumerate(candidates):
            normalised = self._normalize_score(scores[i])
            final = (
                self._rerank_weight * normalised
                + self._rrf_weight * candidate.rrf_score
            )
            reranked.append(
                RankedResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    paper_id=candidate.paper_id,
                    section_type=candidate.section_type,
                    section_title=candidate.section_title,
                    page_numbers=list(candidate.page_numbers),
                    metadata=dict(candidate.metadata),
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    rrf_score=candidate.rrf_score,
                    rerank_score=round(normalised, 4),
                    final_score=round(final, 4),
                )
            )

        reranked.sort(key=lambda r: r.final_score, reverse=True)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, candidates: list[RankedResult]) -> str:
        """Construct the reranking prompt.

        Asks Gemini to rate each passage 1–10 and return JSON.
        """
        passages: list[str] = []
        for i, c in enumerate(candidates, 1):
            truncated = c.text[:_MAX_PASSAGE_CHARS]
            passages.append(f"Passage {i}: {truncated}")

        passages_block = "\n\n".join(passages)
        return (
            "You are a relevance assessor. Rate how relevant each passage "
            "is to the given query.\n\n"
            f"Query: {query}\n\n"
            f"{passages_block}\n\n"
            f"Rate each passage's relevance on a scale of 1-10 "
            f"(1=completely irrelevant, 10=perfectly relevant).\n"
            f"Return ONLY valid JSON in this exact format: "
            f'{{"scores": [score1, score2, ...]}}\n'
            f"You must return exactly {len(candidates)} scores."
        )

    # ------------------------------------------------------------------
    # Gemini API
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with exponential-backoff retry.

        Args:
            prompt: The prompt to send.

        Returns:
            Raw response text.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        self._ensure_api_key()

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(self._model_name)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Reranker attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Reranking failed after {self._max_retries} retries: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_scores(
        self, response_text: str, num_candidates: int
    ) -> list[float]:
        """Parse Gemini's JSON response into a list of scores.

        Handles ``{"scores": [...]}``, bare ``[...]``, and markdown fences.
        Returns empty list on any parse failure.

        Args:
            response_text: Raw text from Gemini.
            num_candidates: Expected number of scores.

        Returns:
            List of scores (1–10), or empty list on failure.
        """
        try:
            cleaned = self._strip_markdown_fences(response_text)
            data = json.loads(cleaned)

            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
            elif isinstance(data, list):
                scores = data
            else:
                logger.warning("Unexpected JSON structure from reranker: %s", type(data))
                return []

            if len(scores) != num_candidates:
                logger.warning(
                    "Score count mismatch: expected %d, got %d",
                    num_candidates,
                    len(scores),
                )
                return []

            # Validate and clamp each score
            result: list[float] = []
            for s in scores:
                val = float(s)
                val = max(1.0, min(10.0, val))
                result.append(val)

            return result

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse reranker response: %s", exc)
            return []

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from text."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Normalise a 1–10 score to 0.0–1.0."""
        return max(0.0, min(1.0, (score - 1.0) / 9.0))

    @staticmethod
    def _ensure_api_key() -> None:
        """Raise immediately if no API key is configured."""
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )
