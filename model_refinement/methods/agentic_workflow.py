"""
AgenticWorkflow: LangGraph workflow with analyzer LLM (optional tool) -> refiner LLM -> params.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ..ml_model import TrainableModel

ROOT = Path(__file__).resolve().parent.parent.parent


class WorkflowState(TypedDict, total=False):
    user_input: str
    training_log: str
    diagnosis: str
    tool_results: List[Dict[str, Any]]
    tool_call_params: List[Dict[str, Any]]
    tool_call_count: int
    analyzer_output: Dict[str, Any]
    refined_params: Dict[str, Any]


MAX_TOOL_CALLS = 3
PARAM_KEYS = [
    "dataset_path", "n_rows", "test_size", "random_state", "metrics",
    "n_estimators", "max_depth", "learning_rate", "min_samples_leaf",
    "min_samples_split", "subsample", "max_features",
]

ANALYZER_SYSTEM = """You are a failure analyzer for a gradient-boosting income-prediction model. You receive the user's problem description and the current training run (config + metrics). You may optionally call a tool to train the model with chosen params and get back metrics and feature importances.

Output a JSON object with:
- "decision": "use_tool" or "no_tool"
- "diagnosis": (required if decision is "no_tool", or after any tool calls) short diagnosis for the next step. If you have already run the tool 1+ times, summarize what the tool results show so the refiner can decide final params.
- "tool_params": (required only if decision is "use_tool") a dict with keys: dataset_path, n_rows, test_size, random_state, metrics, n_estimators, max_depth, learning_rate, min_samples_leaf, min_samples_split, subsample, max_features. Use the same dataset_path and test_size as in the training log; you may set n_rows (int or null) and model params to run a quick experiment.

Reply with only the JSON object, no markdown."""

ANALYZER_USER = """User problem:
{user_input}

Current training run:
{training_log}
{tool_results_block}

Output JSON (decision, diagnosis if not calling tool or to summarize after tool runs, tool_params if decision is use_tool):"""

REFINER_SYSTEM = """You are a model refinement assistant. Given the user's problem, the current training log, and the analyzer's diagnosis (and any tool results summarized there), output a single JSON object with the final refined parameters. Use these keys: dataset_path, n_rows, test_size, random_state, metrics, n_estimators, max_depth, learning_rate, min_samples_leaf, min_samples_split, subsample, max_features. Reply with only the JSON object, no markdown."""

REFINER_USER = """User problem:
{user_input}

Current training run:
{training_log}

Analyzer diagnosis:
{diagnosis}

Output refined params as a single JSON object:"""


def _parse_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _train_tool(tool_params: Dict[str, Any]) -> Dict[str, Any]:
    """Train with given params (subset allowed for speed), return metrics + feature_importances."""
    model = TrainableModel.from_config(tool_params)
    model.train()
    return model.get_tool_result()


class AgenticWorkflow:
    """LangGraph: analyzer (optional tool, max 3 calls) -> refiner -> refined params."""

    def __init__(self, model_name: str = "gpt-4o-mini", max_tool_calls: int = MAX_TOOL_CALLS):
        self.max_tool_calls = max_tool_calls
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYZER_SYSTEM),
            ("human", ANALYZER_USER),
        ])
        self.refiner_prompt = ChatPromptTemplate.from_messages([
            ("system", REFINER_SYSTEM),
            ("human", REFINER_USER),
        ])
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)

        graph.add_node("analyzer", self._analyzer_node)
        graph.add_node("run_tool", self._run_tool_node)
        graph.add_node("refiner", self._refiner_node)

        graph.set_entry_point("analyzer")
        graph.add_conditional_edges("analyzer", self._after_analyzer, {"run_tool": "run_tool", "refiner": "refiner"})
        graph.add_edge("run_tool", "analyzer")
        graph.add_edge("refiner", END)

        return graph.compile()

    def _analyzer_node(self, state: WorkflowState) -> Dict[str, Any]:
        tool_results = state.get("tool_results") or []
        tool_results_block = ""
        if tool_results:
            tool_results_block = "\nPrevious tool results:\n" + json.dumps(tool_results, indent=2)
        msg = self.analyzer_prompt | self.llm
        out = msg.invoke({
            "user_input": state["user_input"],
            "training_log": state["training_log"],
            "tool_results_block": tool_results_block,
        })
        analyzer_output = _parse_json(out.content)
        diagnosis = analyzer_output.get("diagnosis") or state.get("diagnosis") or ""
        return {"analyzer_output": analyzer_output, "diagnosis": diagnosis}

    def _after_analyzer(self, state: WorkflowState) -> Literal["run_tool", "refiner"]:
        out = state.get("analyzer_output") or {}
        decision = out.get("decision", "no_tool")
        count = state.get("tool_call_count") or 0
        if decision == "use_tool" and count < self.max_tool_calls and out.get("tool_params"):
            return "run_tool"
        return "refiner"

    def _run_tool_node(self, state: WorkflowState) -> Dict[str, Any]:
        tool_params = (state.get("analyzer_output") or {}).get("tool_params") or {}
        # Resolve dataset_path relative to project root
        if tool_params.get("dataset_path") and not Path(tool_params["dataset_path"]).is_absolute():
            tool_params = {**tool_params, "dataset_path": str(ROOT / tool_params["dataset_path"])}
        result = _train_tool(tool_params)
        tool_results = list(state.get("tool_results") or [])
        tool_results.append(result)
        tool_call_params = list(state.get("tool_call_params") or [])
        tool_call_params.append(dict(tool_params))
        return {
            "tool_results": tool_results,
            "tool_call_params": tool_call_params,
            "tool_call_count": (state.get("tool_call_count") or 0) + 1,
        }

    def _refiner_node(self, state: WorkflowState) -> Dict[str, Any]:
        msg = self.refiner_prompt | self.llm
        out = msg.invoke({
            "user_input": state["user_input"],
            "training_log": state["training_log"],
            "diagnosis": state.get("diagnosis") or "",
        })
        refined = _parse_json(out.content)
        return {"refined_params": self._normalize_params(refined, state["training_log"])}

    def _normalize_params(self, params: Dict[str, Any], training_log_str: str) -> Dict[str, Any]:
        """Ensure all PARAM_KEYS present; parse training_log if needed for fallbacks."""
        try:
            fallback = json.loads(training_log_str) if training_log_str.strip().startswith("{") else {}
        except json.JSONDecodeError:
            fallback = {}
        out = {}
        for k in PARAM_KEYS:
            v = params.get(k) if k in params else fallback.get(k)
            if k == "n_rows" and v is not None:
                try:
                    v = int(float(v))
                except (TypeError, ValueError):
                    v = fallback.get("n_rows")
            if k == "metrics" and isinstance(v, str):
                v = [s.strip() for s in v.strip("[]").replace("'", "").split(",")] if v else None
            out[k] = v
        return out

    def run(self, user_input: str, training_log_path: str) -> Dict[str, Any]:
        """Load training log CSV, run graph, return refined_params."""
        import pandas as pd
        path = Path(training_log_path)
        if not path.is_absolute():
            path = ROOT / path
        df = pd.read_csv(path)
        if len(df) == 0:
            raise ValueError("Training log CSV is empty")
        training_log = json.dumps({k: (v if pd.notna(v) else None) for k, v in df.iloc[0].to_dict().items()}, indent=2)
        initial: WorkflowState = {
            "user_input": user_input,
            "training_log": training_log,
            "tool_results": [],
            "tool_call_params": [],
            "tool_call_count": 0,
        }
        final = self._graph.invoke(initial)
        return {
            "refined_params": final.get("refined_params") or {},
            "tool_call_count": final.get("tool_call_count") or 0,
            "tool_results": final.get("tool_results") or [],
            "tool_call_params": final.get("tool_call_params") or [],
            "diagnosis": final.get("diagnosis") or "",
        }