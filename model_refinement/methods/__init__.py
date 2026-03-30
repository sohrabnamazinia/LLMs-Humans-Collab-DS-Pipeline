"""Model-refinement strategies (LLM baselines + optional AutoML)."""

from __future__ import annotations

__all__ = ["SingleLLM", "AgenticWorkflow"]


def __getattr__(name: str):
    if name == "SingleLLM":
        from .single_llm import SingleLLM

        return SingleLLM
    if name == "AgenticWorkflow":
        from .agentic_workflow import AgenticWorkflow

        return AgenticWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
