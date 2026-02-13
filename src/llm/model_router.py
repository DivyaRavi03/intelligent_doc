"""Task-based model routing for optimal Gemini model selection.

Routes each task type to the most cost-effective model that meets quality
requirements.  Simple tasks (extraction, QA, summarization) use the fast
model; complex tasks (comparison, evaluation) use the pro model.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    EXTRACTION = "extraction"
    QA = "qa"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    EVALUATION = "evaluation"


_DEFAULT_RULES: dict[TaskType, str] = {
    TaskType.EXTRACTION: "gemini-2.0-flash",
    TaskType.QA: "gemini-2.0-flash",
    TaskType.SUMMARIZATION: "gemini-2.0-flash",
    TaskType.COMPARISON: "gemini-1.5-pro",
    TaskType.EVALUATION: "gemini-1.5-pro",
}


class ModelRouter:
    """Select the optimal Gemini model based on task type.

    Args:
        overrides: Optional mapping of task types to model names that
            replaces the corresponding default routing rules.
    """

    def __init__(self, overrides: dict[TaskType, str] | None = None) -> None:
        self._rules = dict(_DEFAULT_RULES)
        if overrides:
            self._rules.update(overrides)

    def route(self, task: TaskType) -> str:
        """Return the model name for the given task type."""
        return self._rules[task]

    def get_rules(self) -> dict[str, str]:
        """Return current routing rules as ``{task_name: model_name}``."""
        return {t.value: m for t, m in self._rules.items()}
