"""Citation-tracked question-answering engine.

The :class:`QAEngine` answers questions about research papers by retrieving
relevant passages via :class:`HybridRetriever`, prompting Gemini with
source-cited instructions, and verifying each claim against the cited
sources for a faithfulness score.
"""

from __future__ import annotations

import json
import logging
import re

from src.llm.gemini_client import GeminiClient
from src.llm.prompts import QA_ANSWER, QA_VERIFY_CLAIMS
from src.models.schemas import (
    Citation,
    ClaimVerification,
    QAResponse,
)
from src.retrieval.hybrid_retriever import HybridRetriever, RankedResult

logger = logging.getLogger(__name__)


class QAEngine:
    """Question-answering engine with citation tracking and verification.

    Uses :class:`HybridRetriever` to find relevant passages, then prompts
    Gemini to answer with ``[N]`` source citations.  Each claim is verified
    against its cited source for a faithfulness score.

    Args:
        client: :class:`GeminiClient` instance.
        retriever: :class:`HybridRetriever` for context retrieval.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self._client = client or GeminiClient()
        self._retriever = retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> QAResponse:
        """Answer a question using retrieved context passages.

        Args:
            query: The user's question.
            paper_ids: Optional list of paper IDs to restrict search.
            top_k: Number of context passages to retrieve.

        Returns:
            :class:`QAResponse` with answer, citations, verifications,
            and faithfulness score.
        """
        if not query or not query.strip():
            return QAResponse(query=query, answer="")

        chunks = self._retrieve_chunks(query, paper_ids, top_k)
        if not chunks:
            return QAResponse(
                query=query,
                answer="No relevant context found for this query.",
            )

        context, citations = self._build_context(chunks)
        answer_text = self._generate(query, context)
        verifications, faithfulness, flagged = self._verify(
            answer_text, context, citations
        )

        return QAResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            claim_verifications=verifications,
            faithfulness_score=faithfulness,
            flagged_claims=flagged,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _retrieve_chunks(
        self,
        query: str,
        paper_ids: list[str] | None,
        top_k: int,
    ) -> list[RankedResult]:
        """Retrieve relevant chunks, optionally filtered by paper IDs."""
        if self._retriever is None:
            return []

        if not paper_ids:
            return self._retriever.retrieve(query, top_k=top_k)

        # Retrieve per paper, then merge
        all_results: list[RankedResult] = []
        for pid in paper_ids:
            try:
                results = self._retriever.retrieve(
                    query, top_k=top_k, paper_id=pid
                )
                all_results.extend(results)
            except Exception:
                logger.warning("Retrieval failed for paper %s", pid, exc_info=True)

        # Deduplicate and sort by final_score
        seen: set[str] = set()
        unique: list[RankedResult] = []
        for r in sorted(all_results, key=lambda x: x.final_score, reverse=True):
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique.append(r)

        return unique[:top_k]

    def _build_context(
        self,
        chunks: list[RankedResult],
    ) -> tuple[str, list[Citation]]:
        """Format chunks with ``[1]`` ``[2]`` markers.

        Returns:
            ``(context_string, citations_list)``
        """
        parts: list[str] = []
        citations: list[Citation] = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"Source [{i}]:\n{chunk.text}\n")
            citations.append(
                Citation(
                    source_index=i,
                    chunk_id=chunk.chunk_id,
                    text_snippet=chunk.text[:200],
                    paper_id=chunk.paper_id,
                    section_type=chunk.section_type,
                    page_numbers=list(chunk.page_numbers),
                )
            )
        return "\n".join(parts), citations

    def _generate(self, query: str, context: str) -> str:
        """Generate an answer using Gemini."""
        try:
            prompt = QA_ANSWER.format(query=query, context=context)
            response = self._client.generate(prompt)
            # The response may be JSON-wrapped; extract content
            content = response.content
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "answer" in data:
                    return data["answer"]
            except (json.JSONDecodeError, TypeError):
                pass
            return content
        except RuntimeError:
            logger.warning("QA generation failed")
            return "Unable to generate an answer at this time."

    def _verify(
        self,
        answer: str,
        context: str,
        citations: list[Citation],
    ) -> tuple[list[ClaimVerification], float, list[str]]:
        """Verify each claim against its cited source.

        Returns:
            ``(verifications, faithfulness_score, flagged_claims)``
        """
        if not answer or not citations:
            return [], 0.0, []

        # Build sources text
        sources_text = ""
        for c in citations:
            sources_text += f"[{c.source_index}] {c.text_snippet}\n\n"

        try:
            prompt = QA_VERIFY_CLAIMS.format(answer=answer, sources=sources_text)
            response = self._client.generate(prompt)
            verifications = self._parse_verifications(response.content)
        except RuntimeError:
            logger.warning("Verification failed, returning empty")
            return [], 0.0, []

        if not verifications:
            return [], 0.0, []

        # Compute faithfulness score
        supported = sum(
            1 for v in verifications if v.status == "SUPPORTED"
        )
        total = len(verifications)
        faithfulness = supported / total if total > 0 else 0.0

        flagged = [v.claim for v in verifications if v.status == "NOT_SUPPORTED"]

        return verifications, round(faithfulness, 4), flagged

    def _split_claims(
        self, answer: str
    ) -> list[tuple[str, int | None]]:
        """Split answer into ``(claim, cited_source_index)`` tuples.

        Args:
            answer: The generated answer text.

        Returns:
            List of ``(sentence, source_index)`` tuples.  ``source_index``
            is ``None`` for uncited claims.
        """
        if not answer:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        results: list[tuple[str, int | None]] = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            citations = re.findall(r"\[(\d+)\]", sentence)
            if citations:
                results.append((sentence.strip(), int(citations[0])))
            else:
                results.append((sentence.strip(), None))
        return results

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verifications(response_text: str) -> list[ClaimVerification]:
        """Parse Gemini's verification response into ClaimVerification objects."""
        try:
            cleaned = QAEngine._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            raw_list = []
            if isinstance(data, dict) and "verifications" in data:
                raw_list = data["verifications"]
            elif isinstance(data, list):
                raw_list = data

            verifications: list[ClaimVerification] = []
            for item in raw_list:
                verifications.append(
                    ClaimVerification(
                        claim=item.get("claim", ""),
                        cited_source_index=item.get("cited_source_index"),
                        status=item.get("status", "NOT_SUPPORTED"),
                        explanation=item.get("explanation", ""),
                    )
                )
            return verifications
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
