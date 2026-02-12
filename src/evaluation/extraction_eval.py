"""Extraction evaluation with exact match and ROUGE-L metrics.

The :class:`ExtractionEvaluator` compares LLM extractions against
ground-truth metadata, using exact match for structured fields and
ROUGE-L for free-text fields.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.extractor import PaperExtractor
from src.models.schemas import DocumentMetadataSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FieldEvalResult:
    """Evaluation result for a single field."""

    field_name: str
    score: float  # 0.0 to 1.0
    metric: str  # "exact_match", "rouge_l", "precision_recall"
    details: str = ""


@dataclass
class PaperEvalResult:
    """Per-paper evaluation result."""

    paper_id: str
    field_scores: list[FieldEvalResult] = field(default_factory=list)
    overall_score: float = 0.0


@dataclass
class ExtractionEvalResult:
    """Aggregated extraction evaluation result."""

    num_papers: int = 0
    per_field_accuracy: dict[str, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    per_paper: list[PaperEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "num_papers": self.num_papers,
            "per_field_accuracy": self.per_field_accuracy,
            "overall_accuracy": self.overall_accuracy,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ExtractionEvaluator:
    """Evaluate extraction quality against ground-truth test sets."""

    def evaluate(
        self,
        test_set: list[dict],
        extractor: PaperExtractor,
    ) -> ExtractionEvalResult:
        """Run evaluation on test papers.

        For each paper the evaluator:
        1. Calls ``extractor.extract(paper_text)`` for findings.
        2. Parses title/authors from the paper text header.
        3. Compares extracted values against expected ground truth.

        Args:
            test_set: List of dicts with ``paper_text`` and expected fields.
            extractor: The extractor to evaluate.

        Returns:
            Aggregated accuracy and per-paper breakdown.
        """
        if not test_set:
            return ExtractionEvalResult()

        per_paper: list[PaperEvalResult] = []
        field_totals: dict[str, list[float]] = {}

        for entry in test_set:
            paper_id = entry["paper_id"]
            paper_text = entry["paper_text"]
            scores: list[FieldEvalResult] = []

            # --- Run extraction ---
            try:
                metadata = DocumentMetadataSchema(
                    title=entry.get("expected_title", ""),
                    authors=entry.get("expected_authors", []),
                    confidence=0.9,
                )
                extraction = extractor.extract(paper_text, metadata=metadata)
            except Exception:
                logger.warning("Extraction failed for paper: %s", paper_id)
                paper_result = PaperEvalResult(paper_id=paper_id, overall_score=0.0)
                per_paper.append(paper_result)
                continue

            # --- Title (exact match) ---
            predicted_title = extraction.paper_id  # set from metadata.title
            expected_title = entry.get("expected_title", "")
            title_score = self._exact_match(predicted_title, expected_title)
            scores.append(
                FieldEvalResult("title", title_score, "exact_match")
            )

            # --- Authors (exact match list) ---
            # Parse from second line of paper_text
            lines = [ln.strip() for ln in paper_text.split("\n") if ln.strip()]
            predicted_authors = (
                [a.strip() for a in lines[1].split(",")] if len(lines) > 1 else []
            )
            expected_authors = entry.get("expected_authors", [])
            authors_score = self._exact_match_list(predicted_authors, expected_authors)
            scores.append(
                FieldEvalResult("authors", authors_score, "exact_match")
            )

            # --- Abstract (ROUGE-L) ---
            # Extract abstract from paper_text between "Abstract" and next section
            predicted_abstract = self._extract_section(paper_text, "Abstract")
            expected_abstract = entry.get("expected_abstract", "")
            if predicted_abstract and expected_abstract:
                abstract_score = self._rouge_l(predicted_abstract, expected_abstract)
            elif not expected_abstract:
                abstract_score = 1.0
            else:
                abstract_score = 0.0
            scores.append(
                FieldEvalResult("abstract", abstract_score, "rouge_l")
            )

            # --- Keywords (precision/recall) ---
            predicted_keywords = self._extract_keywords(paper_text)
            expected_keywords = entry.get("expected_keywords", [])
            if expected_keywords:
                prec, rec = self._keyword_precision_recall(
                    predicted_keywords, expected_keywords
                )
                kw_score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            else:
                kw_score = 1.0 if not predicted_keywords else 0.0
            scores.append(
                FieldEvalResult("keywords", kw_score, "precision_recall")
            )

            # --- Findings (ROUGE-L) ---
            predicted_findings = [f.claim for f in extraction.key_findings]
            expected_findings = entry.get("expected_findings", [])
            if predicted_findings and expected_findings:
                pred_text = " ".join(predicted_findings)
                exp_text = " ".join(expected_findings)
                findings_score = self._rouge_l(pred_text, exp_text)
            elif not expected_findings:
                findings_score = 1.0
            else:
                findings_score = 0.0
            scores.append(
                FieldEvalResult("findings", findings_score, "rouge_l")
            )

            # --- Aggregate per-paper ---
            overall = sum(s.score for s in scores) / len(scores) if scores else 0.0
            paper_result = PaperEvalResult(
                paper_id=paper_id, field_scores=scores, overall_score=overall
            )
            per_paper.append(paper_result)

            for s in scores:
                field_totals.setdefault(s.field_name, []).append(s.score)

        # Aggregate per-field
        per_field_accuracy = {
            name: sum(vals) / len(vals) for name, vals in field_totals.items()
        }
        overall_accuracy = (
            sum(per_field_accuracy.values()) / len(per_field_accuracy)
            if per_field_accuracy
            else 0.0
        )

        return ExtractionEvalResult(
            num_papers=len(per_paper),
            per_field_accuracy=per_field_accuracy,
            overall_accuracy=overall_accuracy,
            per_paper=per_paper,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_test_set(path: str | Path) -> list[dict]:
        """Load a test set from a JSON file."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _exact_match(predicted: str, expected: str) -> float:
        """Case-insensitive exact match.  Returns 1.0 or 0.0."""
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0

    @staticmethod
    def _exact_match_list(predicted: list[str], expected: list[str]) -> float:
        """Exact match for lists (order-insensitive)."""
        pred_set = {s.strip().lower() for s in predicted}
        exp_set = {s.strip().lower() for s in expected}
        return 1.0 if pred_set == exp_set else 0.0

    @staticmethod
    def _rouge_l(predicted: str, expected: str) -> float:
        """Compute ROUGE-L F1 score."""
        if not predicted.strip() or not expected.strip():
            return 0.0
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(expected, predicted)
        return scores["rougeL"].fmeasure

    @staticmethod
    def _keyword_precision_recall(
        predicted: list[str], expected: list[str]
    ) -> tuple[float, float]:
        """Compute precision and recall for keyword lists."""
        pred_set = {k.strip().lower() for k in predicted}
        exp_set = {k.strip().lower() for k in expected}
        if not pred_set or not exp_set:
            return 0.0, 0.0
        tp = len(pred_set & exp_set)
        precision = tp / len(pred_set)
        recall = tp / len(exp_set)
        return precision, recall

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """Extract text between a section header and the next section."""
        lines = text.split("\n")
        capturing = False
        section_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == section_name.lower():
                capturing = True
                continue
            if capturing:
                # Stop at next section header (starts with digit or Keywords)
                if (
                    stripped
                    and (stripped[0].isdigit() or stripped.startswith("Keywords"))
                ):
                    break
                section_lines.append(stripped)
        return " ".join(ln for ln in section_lines if ln)

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract keywords from a 'Keywords:' line."""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("keywords:"):
                kw_text = stripped.split(":", 1)[1].strip()
                return [k.strip() for k in kw_text.split(",") if k.strip()]
        return []
