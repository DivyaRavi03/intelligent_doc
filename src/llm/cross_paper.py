"""Cross-paper comparison using structured prompts.

The :class:`CrossPaperAnalyzer` gathers context from multiple papers,
builds a comparison prompt, and parses the LLM response into a
structured :class:`CrossPaperComparison`.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

from src.llm.prompts import CROSS_PAPER_COMPARE
from src.models.schemas import ComparisonTableRow, CrossPaperComparison

if TYPE_CHECKING:
    from src.api.stores import InMemoryDocumentStore
    from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class CrossPaperAnalyzer:
    """Compare multiple research papers on a given aspect.

    Args:
        client: :class:`GeminiClient` for LLM calls.
        store: Document store for retrieving paper content.
    """

    def __init__(self, client: GeminiClient, store: InMemoryDocumentStore) -> None:
        self._client = client
        self._store = store

    def compare_papers(
        self,
        paper_ids: list[str],
        aspect: str = "methodology",
    ) -> CrossPaperComparison:
        """Generate a structured comparison of multiple papers.

        Args:
            paper_ids: List of paper UUIDs to compare (2-5).
            aspect: The aspect to compare (e.g. methodology, results).

        Returns:
            Structured comparison with table, agreements, contradictions.

        Raises:
            ValueError: If fewer than 2 paper IDs are provided.
        """
        if len(paper_ids) < 2:
            raise ValueError("At least 2 paper IDs are required for comparison")

        context = self._gather_context(paper_ids)
        papers_context = self._format_context(context)
        prompt = CROSS_PAPER_COMPARE.format(
            aspect=aspect, papers_context=papers_context
        )

        response = self._client.generate(prompt)
        return self._parse_comparison(
            response.content, paper_ids, aspect, response.model
        )

    def _gather_context(self, paper_ids: list[str]) -> dict[str, str]:
        """Retrieve text content for each paper from the store."""
        context: dict[str, str] = {}
        for paper_id in paper_ids:
            try:
                doc_uuid = uuid.UUID(paper_id)
                doc = self._store.get(doc_uuid)
                if doc and doc.sections:
                    text = "\n".join(s.text[:1000] for s in doc.sections[:3])
                    context[paper_id] = text
                else:
                    context[paper_id] = "Content unavailable"
            except (ValueError, Exception):
                context[paper_id] = "Content unavailable"
        return context

    @staticmethod
    def _format_context(context: dict[str, str]) -> str:
        """Format paper contexts into a numbered block for the prompt."""
        parts: list[str] = []
        for i, (pid, text) in enumerate(context.items(), 1):
            parts.append(f"Paper {i} ({pid}):\n{text[:2000]}")
        return "\n\n".join(parts)

    def _parse_comparison(
        self,
        raw: str,
        paper_ids: list[str],
        aspect: str,
        model: str = "",
    ) -> CrossPaperComparison:
        """Parse LLM response into a structured comparison."""
        cleaned = self._strip_markdown_fences(raw)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse comparison JSON, using raw response")
            return CrossPaperComparison(
                paper_ids=paper_ids,
                aspect=aspect,
                synthesis=raw,
                model_used=model,
            )

        table_rows = [
            ComparisonTableRow(
                aspect=row.get("aspect", ""),
                papers=row.get("papers", {}),
            )
            for row in data.get("comparison_table", [])
        ]

        return CrossPaperComparison(
            paper_ids=paper_ids,
            aspect=aspect,
            comparison_table=table_rows,
            agreements=data.get("agreements", []),
            contradictions=data.get("contradictions", []),
            synthesis=data.get("synthesis", ""),
            model_used=model,
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (``\\`\\`\\`json ... \\`\\`\\``)."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
