"""Structured paper extraction with dual-prompt confidence scoring.

The :class:`PaperExtractor` extracts key findings, methodology, and
quantitative results from research papers.  It uses a dual-prompt
strategy (two differently-framed prompts) and fuzzy-match source grounding
to compute confidence scores.  Extractions below the confidence threshold
are flagged for human review.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from difflib import SequenceMatcher

from src.llm.gemini_client import GeminiClient
from src.llm.prompts import (
    EXTRACT_KEY_FINDINGS,
    EXTRACT_KEY_FINDINGS_ALT,
    EXTRACT_METHODOLOGY,
    EXTRACT_RESULTS,
)
from src.models.schemas import (
    DocumentMetadataSchema,
    Finding,
    MethodologyExtraction,
    PaperExtraction,
    ResultExtraction,
)

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.7
_FUZZY_MATCH_THRESHOLD = 0.6


class PaperExtractor:
    """Extract structured information from research papers.

    Uses dual-prompt extraction and fuzzy-match source grounding to
    compute confidence scores.  Extractions with confidence < 0.7 are
    flagged as ``needs_review``.

    Args:
        client: :class:`GeminiClient` instance.  Creates one if not
            provided.
    """

    def __init__(self, client: GeminiClient | None = None) -> None:
        self._client = client or GeminiClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        paper_text: str,
        metadata: DocumentMetadataSchema | None = None,
    ) -> PaperExtraction:
        """Run the full extraction pipeline.

        Args:
            paper_text: Full text of the paper.
            metadata: Optional document metadata for paper_id.

        Returns:
            :class:`PaperExtraction` with findings, methodology, results,
            and overall confidence.
        """
        paper_id = (metadata.title if metadata and metadata.title else "unknown")

        findings = self.extract_key_findings(paper_text)
        methodology = self.extract_methodology(paper_text)
        results = self.extract_results(paper_text)

        # Overall confidence — average of component confidences
        scores: list[float] = []
        if findings:
            scores.append(sum(f.confidence for f in findings) / len(findings))
        if methodology:
            scores.append(0.8)  # methodology has no per-item confidence
        if results:
            scores.append(0.8)

        overall = sum(scores) / len(scores) if scores else 0.0

        return PaperExtraction(
            paper_id=paper_id,
            key_findings=findings,
            methodology=methodology,
            results=results,
            confidence=round(overall, 4),
            needs_review=overall < _CONFIDENCE_THRESHOLD,
        )

    def extract_key_findings(self, text: str) -> list[Finding]:
        """Extract key findings with dual-prompt confidence scoring.

        Args:
            text: Paper text to extract findings from.

        Returns:
            List of :class:`Finding` objects with confidence scores.
        """
        if not text.strip():
            return []

        try:
            findings_raw, consistency = self._dual_prompt_extract(
                text,
                EXTRACT_KEY_FINDINGS,
                EXTRACT_KEY_FINDINGS_ALT,
                self._parse_findings,
            )
        except RuntimeError:
            logger.warning("Key findings extraction failed")
            return []

        if not findings_raw:
            return []

        # Check source grounding and compute final confidence
        results: list[Finding] = []
        grounding_scores: list[float] = []
        for f in findings_raw:
            grounding = self._check_source_grounding(
                f.get("supporting_quote", ""), text
            )
            grounding_scores.append(grounding)

            completeness = min(1.0, len(findings_raw) / 3.0)
            confidence = self._compute_confidence(
                consistency, grounding_scores, completeness
            )

            results.append(
                Finding(
                    claim=f.get("claim", ""),
                    supporting_quote=f.get("supporting_quote", ""),
                    confidence=round(min(1.0, max(0.0, confidence)), 4),
                )
            )

        return results

    def extract_methodology(self, text: str) -> MethodologyExtraction | None:
        """Extract methodology details.

        Args:
            text: Paper text.

        Returns:
            :class:`MethodologyExtraction` or ``None`` on failure.
        """
        if not text.strip():
            return None

        try:
            response = self._client.generate(EXTRACT_METHODOLOGY.format(text=text[:8000]))
            data = self._parse_methodology(response.content)
            if data is None:
                return None
            return MethodologyExtraction(
                approach=data.get("approach", ""),
                datasets=data.get("datasets", []),
                tools=data.get("tools", []),
                eval_metrics=data.get("eval_metrics", []),
            )
        except RuntimeError:
            logger.warning("Methodology extraction failed")
            return None

    def extract_results(self, text: str) -> list[ResultExtraction]:
        """Extract quantitative results.

        Args:
            text: Paper text.

        Returns:
            List of :class:`ResultExtraction` objects.
        """
        if not text.strip():
            return []

        try:
            response = self._client.generate(EXTRACT_RESULTS.format(text=text[:8000]))
            raw_list = self._parse_results(response.content)
        except RuntimeError:
            logger.warning("Results extraction failed")
            return []

        results: list[ResultExtraction] = []
        for r in raw_list:
            results.append(
                ResultExtraction(
                    metric_name=r.get("metric_name", ""),
                    value=r.get("value", ""),
                    baseline=r.get("baseline"),
                    improvement=r.get("improvement"),
                    table_reference=r.get("table_reference"),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _dual_prompt_extract(
        self,
        text: str,
        prompt_a: str,
        prompt_b: str,
        parse_func: Callable[[str], list[dict]],
    ) -> tuple[list[dict], float]:
        """Run extraction with two prompts and compute consistency.

        Args:
            text: Source text.
            prompt_a: Primary prompt template.
            prompt_b: Alternative prompt template.
            parse_func: Function to parse response into list of dicts.

        Returns:
            ``(primary_results, consistency_score)``
        """
        resp_a = self._client.generate(prompt_a.format(text=text[:8000]))
        resp_b = self._client.generate(prompt_b.format(text=text[:8000]))

        items_a = parse_func(resp_a.content)
        items_b = parse_func(resp_b.content)

        if not items_a:
            return [], 0.0

        # Compute consistency — how many claims from A appear in B
        claims_a = {d.get("claim", "").lower().strip() for d in items_a}
        claims_b = {d.get("claim", "").lower().strip() for d in items_b}

        if not claims_a:
            return items_a, 0.0

        overlap = 0
        for claim_a in claims_a:
            for claim_b in claims_b:
                ratio = SequenceMatcher(None, claim_a, claim_b).ratio()
                if ratio >= _FUZZY_MATCH_THRESHOLD:
                    overlap += 1
                    break

        consistency = overlap / max(len(claims_a), len(claims_b)) if claims_b else 0.0
        return items_a, round(consistency, 4)

    @staticmethod
    def _check_source_grounding(quote: str, source_text: str) -> float:
        """Check how well a supporting quote matches the source text.

        Uses ``difflib.SequenceMatcher`` for fuzzy matching and a
        substring check as a fast path.

        Args:
            quote: The supporting quote to verify.
            source_text: The original paper text.

        Returns:
            Grounding score in ``[0.0, 1.0]``.
        """
        if not quote or not source_text:
            return 0.0

        q_lower = quote.lower()
        s_lower = source_text.lower()

        # Exact substring match is high confidence
        if q_lower in s_lower:
            return 1.0

        # Fuzzy match
        ratio = SequenceMatcher(None, q_lower, s_lower).ratio()
        return round(min(1.0, ratio), 4)

    @staticmethod
    def _compute_confidence(
        consistency_score: float,
        grounding_scores: list[float],
        completeness_score: float,
    ) -> float:
        """Weighted confidence: 0.4*consistency + 0.4*grounding + 0.2*completeness."""
        avg_grounding = (
            sum(grounding_scores) / len(grounding_scores)
            if grounding_scores
            else 0.0
        )
        return (
            0.4 * consistency_score
            + 0.4 * avg_grounding
            + 0.2 * completeness_score
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_findings(response_text: str) -> list[dict]:
        """Parse findings JSON response."""
        try:
            cleaned = PaperExtractor._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            if isinstance(data, dict) and "findings" in data:
                return data["findings"]
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _parse_methodology(response_text: str) -> dict | None:
        """Parse methodology JSON response."""
        try:
            cleaned = PaperExtractor._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            if isinstance(data, dict) and "approach" in data:
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _parse_results(response_text: str) -> list[dict]:
        """Parse results JSON response."""
        try:
            cleaned = PaperExtractor._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (``\\`\\`\\`json ... \\`\\`\\``)."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
