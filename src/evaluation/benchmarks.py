"""Benchmark suite that orchestrates all evaluators and generates reports.

The :class:`BenchmarkSuite` runs retrieval, extraction, and QA evaluators,
compares results against quality thresholds, and produces a JSON report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.extraction_eval import ExtractionEvaluator, ExtractionEvalResult
from src.evaluation.qa_eval import QAEvaluator, QAEvalResult
from src.evaluation.retrieval_eval import RetrievalEvaluator, RetrievalEvalResult
from src.llm.extractor import PaperExtractor
from src.llm.qa_engine import QAEngine
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Pass/fail quality thresholds
THRESHOLDS: dict[str, float] = {
    "retrieval_precision_at_5": 0.75,
    "extraction_accuracy": 0.80,
    "qa_faithfulness": 3.5,  # out of 5.0
}

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    timestamp: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    previous_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "timestamp": self.timestamp,
            "retrieval": self.retrieval,
            "extraction": self.extraction,
            "qa": self.qa,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "previous_report": self.previous_report,
        }


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Run all evaluators and produce a benchmark report.

    Args:
        retrieval_evaluator: Optional custom :class:`RetrievalEvaluator`.
        extraction_evaluator: Optional custom :class:`ExtractionEvaluator`.
        qa_evaluator: Optional custom :class:`QAEvaluator`.
        retrieval_test_set_path: Path to retrieval test set JSON.
        extraction_test_set_path: Path to extraction test set JSON.
    """

    def __init__(
        self,
        retrieval_evaluator: RetrievalEvaluator | None = None,
        extraction_evaluator: ExtractionEvaluator | None = None,
        qa_evaluator: QAEvaluator | None = None,
        retrieval_test_set_path: Path | None = None,
        extraction_test_set_path: Path | None = None,
    ) -> None:
        self._retrieval_eval = retrieval_evaluator or RetrievalEvaluator()
        self._extraction_eval = extraction_evaluator or ExtractionEvaluator()
        self._qa_eval = qa_evaluator or QAEvaluator()
        self._retrieval_test_path = retrieval_test_set_path or (
            _FIXTURES_DIR / "retrieval_test_set.json"
        )
        self._extraction_test_path = extraction_test_set_path or (
            _FIXTURES_DIR / "extraction_test_set.json"
        )

    def run_all(
        self,
        retriever: HybridRetriever,
        extractor: PaperExtractor,
        qa_engine: QAEngine,
        output_path: str | Path | None = None,
    ) -> BenchmarkReport:
        """Run all evaluations and produce a report.

        Args:
            retriever: The retriever to evaluate.
            extractor: The extractor to evaluate.
            qa_engine: The QA engine to evaluate.
            output_path: Optional path to write the JSON report.

        Returns:
            :class:`BenchmarkReport` with all results and pass/fail status.
        """
        report = BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            thresholds=dict(THRESHOLDS),
        )

        # --- Load test sets ---
        retrieval_test_set = RetrievalEvaluator.load_test_set(
            self._retrieval_test_path
        )
        extraction_test_set = ExtractionEvaluator.load_test_set(
            self._extraction_test_path
        )

        # --- Run retrieval evaluation ---
        logger.info("Running retrieval evaluation (%d queries)...", len(retrieval_test_set))
        try:
            retrieval_result: RetrievalEvalResult = self._retrieval_eval.evaluate(
                retrieval_test_set, retriever
            )
            report.retrieval = retrieval_result.to_dict()
        except Exception:
            logger.exception("Retrieval evaluation failed")
            report.retrieval = {"error": "evaluation failed"}

        # --- Run extraction evaluation ---
        logger.info("Running extraction evaluation (%d papers)...", len(extraction_test_set))
        try:
            extraction_result: ExtractionEvalResult = self._extraction_eval.evaluate(
                extraction_test_set, extractor
            )
            report.extraction = extraction_result.to_dict()
        except Exception:
            logger.exception("Extraction evaluation failed")
            report.extraction = {"error": "evaluation failed"}

        # --- Run QA evaluation (reuses retrieval test set for queries) ---
        logger.info("Running QA evaluation (%d queries)...", len(retrieval_test_set))
        try:
            qa_result: QAEvalResult = self._qa_eval.evaluate(
                retrieval_test_set, qa_engine
            )
            report.qa = qa_result.to_dict()
        except Exception:
            logger.exception("QA evaluation failed")
            report.qa = {"error": "evaluation failed"}

        # --- Check thresholds ---
        self._check_thresholds(report)

        # --- Update Prometheus gauges ---
        try:
            from src.monitoring.metrics import update_eval_gauges

            precision = report.retrieval.get("precision_at_k", {}).get("5", 0.0)
            accuracy = report.extraction.get("overall_accuracy", 0.0)
            update_eval_gauges(precision, accuracy)
        except Exception:
            logger.debug("Could not update Prometheus gauges")

        # --- Update admin endpoint ---
        try:
            from src.api.routes_admin import set_latest_evaluation
            from src.models.schemas import EvaluationResult

            eval_result = EvaluationResult(
                timestamp=datetime.now(timezone.utc),
                test_set_name="benchmark",
                accuracy=report.extraction.get("overall_accuracy", 0.0),
                faithfulness_score=report.qa.get("avg_faithfulness", 0.0),
                avg_latency_ms=0.0,
                pass_rate=1.0 if report.passed else 0.0,
            )
            set_latest_evaluation(eval_result)
        except Exception:
            logger.debug("Could not update admin evaluation endpoint")

        # --- Write report ---
        if output_path:
            out = Path(output_path)
            # Load previous report for comparison
            if out.exists():
                try:
                    with open(out) as f:
                        report.previous_report = json.load(f)
                except Exception:
                    pass

            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info("Report written to %s", out)

        return report

    @staticmethod
    def _check_thresholds(report: BenchmarkReport) -> None:
        """Check metrics against thresholds and populate pass/fail."""
        failures: list[str] = []

        # Retrieval precision@5
        retrieval_p5 = report.retrieval.get("precision_at_k", {}).get("5", 0.0)
        threshold_p5 = THRESHOLDS["retrieval_precision_at_5"]
        if retrieval_p5 < threshold_p5:
            failures.append(
                f"retrieval_precision@5 = {retrieval_p5:.3f} "
                f"(threshold: {threshold_p5})"
            )

        # Extraction accuracy
        ext_acc = report.extraction.get("overall_accuracy", 0.0)
        threshold_ext = THRESHOLDS["extraction_accuracy"]
        if ext_acc < threshold_ext:
            failures.append(
                f"extraction_accuracy = {ext_acc:.3f} "
                f"(threshold: {threshold_ext})"
            )

        # QA faithfulness
        qa_faith = report.qa.get("avg_faithfulness", 0.0)
        threshold_qa = THRESHOLDS["qa_faithfulness"]
        if qa_faith < threshold_qa:
            failures.append(
                f"qa_faithfulness = {qa_faith:.2f}/5 "
                f"(threshold: {threshold_qa})"
            )

        report.failures = failures
        report.passed = len(failures) == 0
