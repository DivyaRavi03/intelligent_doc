#!/usr/bin/env python3
"""CLI script to run the full evaluation benchmark suite.

Usage::

    python scripts/run_eval.py --output reports/eval_20260212.json

Exits non-zero if any metric falls below the quality threshold,
making it suitable for CI/CD integration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.benchmarks import THRESHOLDS, BenchmarkSuite  # noqa: E402
from src.evaluation.extraction_eval import ExtractionEvaluator  # noqa: E402
from src.evaluation.qa_eval import QAEvaluator  # noqa: E402
from src.evaluation.retrieval_eval import RetrievalEvaluator  # noqa: E402
from src.llm.extractor import PaperExtractor  # noqa: E402
from src.llm.gemini_client import GeminiClient  # noqa: E402
from src.llm.qa_engine import QAEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_summary(report) -> None:  # noqa: ANN001
    """Print a human-readable summary of the benchmark report."""
    print()
    print("=" * 60)
    print("  EVALUATION BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Timestamp: {report.timestamp}")
    print()

    # Retrieval metrics
    print("  RETRIEVAL METRICS")
    print("  " + "-" * 40)
    precision = report.retrieval.get("precision_at_k", {})
    recall = report.retrieval.get("recall_at_k", {})
    ndcg = report.retrieval.get("ndcg_at_k", {})
    mrr = report.retrieval.get("mrr", 0.0)
    for k in ["1", "3", "5", "10"]:
        p = precision.get(k, 0.0)
        r = recall.get(k, 0.0)
        n = ndcg.get(k, 0.0)
        print(f"  P@{k:>2}: {p:.3f}  |  R@{k:>2}: {r:.3f}  |  NDCG@{k:>2}: {n:.3f}")
    print(f"  MRR:  {mrr:.3f}")
    print()

    # Extraction metrics
    print("  EXTRACTION METRICS")
    print("  " + "-" * 40)
    per_field = report.extraction.get("per_field_accuracy", {})
    for field_name, score in per_field.items():
        print(f"  {field_name:>12}: {score:.3f}")
    overall = report.extraction.get("overall_accuracy", 0.0)
    print(f"  {'overall':>12}: {overall:.3f}")
    print()

    # QA metrics
    print("  QA METRICS")
    print("  " + "-" * 40)
    print(f"  Faithfulness:      {report.qa.get('avg_faithfulness', 0.0):.2f}/5")
    print(f"  Relevance:         {report.qa.get('avg_relevance', 0.0):.2f}/5")
    print(f"  Completeness:      {report.qa.get('avg_completeness', 0.0):.2f}/5")
    print(f"  Answer rate:       {report.qa.get('answer_rate', 0.0):.1%}")
    print(f"  Citation rate:     {report.qa.get('citation_rate', 0.0):.1%}")
    print(f"  Hallucination rate:{report.qa.get('hallucination_rate', 0.0):.1%}")
    print()

    # Pass/fail
    print("  THRESHOLDS")
    print("  " + "-" * 40)
    for name, threshold in THRESHOLDS.items():
        print(f"  {name}: {threshold}")
    print()

    if report.passed:
        print("  RESULT: PASSED")
    else:
        print("  RESULT: FAILED")
        for failure in report.failures:
            print(f"    - {failure}")
    print("=" * 60)


def main() -> int:
    """Run the benchmark suite and return exit code."""
    parser = argparse.ArgumentParser(
        description="Run the evaluation benchmark suite"
    )
    parser.add_argument(
        "--output",
        default=f"reports/eval_{datetime.now().strftime('%Y%m%d')}.json",
        help="Output path for the JSON report (default: reports/eval_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Set up components ---
    try:
        client = GeminiClient()
        extractor = PaperExtractor(client=client)
        qa_engine = QAEngine(client=client)
    except Exception as exc:
        logger.error("Failed to initialise LLM components: %s", exc)
        logger.info("Ensure GEMINI_API_KEY is set in environment.")
        return 1

    # --- Set up retriever ---
    # The retriever requires VectorStore + EmbeddingService + BM25Index
    # with actual indexed data.  For a standalone eval run, the user must
    # have indexed documents first.
    try:
        from src.embedding.embedding_service import EmbeddingService
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.vector_store import VectorStore

        vector_store = VectorStore()
        embedding_service = EmbeddingService()
        bm25_index = BM25Index()
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            bm25_index=bm25_index,
        )
    except Exception as exc:
        logger.error("Failed to initialise retriever: %s", exc)
        return 1

    # --- Run benchmark ---
    suite = BenchmarkSuite(
        retrieval_evaluator=RetrievalEvaluator(),
        extraction_evaluator=ExtractionEvaluator(),
        qa_evaluator=QAEvaluator(judge_client=client),
    )

    logger.info("Running evaluation benchmark suite...")
    report = suite.run_all(retriever, extractor, qa_engine, output_path)

    _print_summary(report)
    logger.info("Report saved to: %s", output_path)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
