"""QA evaluation using LLM-as-Judge for faithfulness, relevance, and completeness.

The :class:`QAEvaluator` sends QA answers to Gemini for scoring on
a 1-5 rubric across three dimensions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.gemini_client import GeminiClient
from src.llm.qa_engine import QAEngine

logger = logging.getLogger(__name__)

_NO_ANSWER_PHRASES = frozenset({
    "unable to generate",
    "no relevant context",
    "i don't know",
    "i cannot answer",
})

QA_JUDGE_PROMPT = """\
You are an expert evaluator for a research paper question-answering system.
Score the following answer on three dimensions.

Question: {query}

Context (ground truth): {context}

Generated Answer: {answer}

Score each dimension from 1 to 5:
- faithfulness: Are all claims in the answer supported by the context? \
(1 = heavily hallucinated, 5 = fully grounded in context)
- relevance: Does the answer directly address the question asked? \
(1 = completely off-topic, 5 = directly and fully answers the question)
- completeness: Does the answer cover all important aspects of the question? \
(1 = missing everything important, 5 = comprehensive coverage)

Return ONLY valid JSON with no extra text:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>, \
"explanation": "<brief reasoning>"}}\
"""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QAQueryEvalResult:
    """Per-query QA evaluation result."""

    query: str
    answer: str = ""
    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    has_answer: bool = False
    has_citations: bool = False
    is_hallucinated: bool = False
    explanation: str = ""


@dataclass
class QAEvalResult:
    """Aggregated QA evaluation result."""

    num_queries: int = 0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    answer_rate: float = 0.0
    citation_rate: float = 0.0
    hallucination_rate: float = 0.0
    per_query: list[QAQueryEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "num_queries": self.num_queries,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevance": self.avg_relevance,
            "avg_completeness": self.avg_completeness,
            "answer_rate": self.answer_rate,
            "citation_rate": self.citation_rate,
            "hallucination_rate": self.hallucination_rate,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class QAEvaluator:
    """Evaluate QA quality using LLM-as-Judge.

    Args:
        judge_client: :class:`GeminiClient` used as the judge LLM.
            Defaults to a new instance.
    """

    def __init__(self, judge_client: GeminiClient | None = None) -> None:
        self._judge = judge_client or GeminiClient()

    def evaluate(
        self,
        test_set: list[dict],
        qa_engine: QAEngine,
    ) -> QAEvalResult:
        """Run QA evaluation.

        Args:
            test_set: List of dicts with ``query``, ``paper_id``,
                ``expected_answer``.
            qa_engine: The QA engine to evaluate.

        Returns:
            Aggregated QA evaluation metrics.
        """
        if not test_set:
            return QAEvalResult()

        per_query: list[QAQueryEvalResult] = []

        for entry in test_set:
            query = entry["query"]
            paper_id = entry.get("paper_id")
            expected = entry.get("expected_answer", "")

            # Run QA engine
            try:
                paper_ids = [paper_id] if paper_id else None
                response = qa_engine.answer(query, paper_ids=paper_ids)
            except Exception:
                logger.warning("QA engine failed for query: %s", query)
                per_query.append(QAQueryEvalResult(query=query))
                continue

            answer = response.answer

            # Determine has_answer
            has_answer = bool(
                answer.strip()
                and not any(p in answer.lower() for p in _NO_ANSWER_PHRASES)
            )

            # Determine has_citations
            has_citations = len(response.citations) > 0

            # Judge the answer
            faith, relev, compl, explanation = self._judge_answer(
                query, answer, expected
            )

            # Determine hallucination (from the QA engine's own score)
            is_hallucinated = response.faithfulness_score < 0.5

            q_result = QAQueryEvalResult(
                query=query,
                answer=answer,
                faithfulness=faith,
                relevance=relev,
                completeness=compl,
                has_answer=has_answer,
                has_citations=has_citations,
                is_hallucinated=is_hallucinated,
                explanation=explanation,
            )
            per_query.append(q_result)

        # Aggregate
        n = len(per_query)
        answered = [q for q in per_query if q.has_answer]

        return QAEvalResult(
            num_queries=n,
            avg_faithfulness=sum(q.faithfulness for q in per_query) / n if n else 0.0,
            avg_relevance=sum(q.relevance for q in per_query) / n if n else 0.0,
            avg_completeness=sum(q.completeness for q in per_query) / n if n else 0.0,
            answer_rate=len(answered) / n if n else 0.0,
            citation_rate=(
                sum(1 for q in per_query if q.has_citations) / n if n else 0.0
            ),
            hallucination_rate=(
                sum(1 for q in per_query if q.is_hallucinated) / n if n else 0.0
            ),
            per_query=per_query,
        )

    def _judge_answer(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> tuple[float, float, float, str]:
        """Send an answer to the LLM judge for scoring.

        Returns:
            ``(faithfulness, relevance, completeness, explanation)``
        """
        prompt = QA_JUDGE_PROMPT.format(
            query=query, context=context, answer=answer
        )
        try:
            response = self._judge.generate(prompt)
            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            data = json.loads(content)
            return (
                float(data.get("faithfulness", 0)),
                float(data.get("relevance", 0)),
                float(data.get("completeness", 0)),
                data.get("explanation", ""),
            )
        except (RuntimeError, json.JSONDecodeError, TypeError, KeyError):
            logger.warning("Judge evaluation failed for query: %s", query)
            return 0.0, 0.0, 0.0, "Judge evaluation failed"
