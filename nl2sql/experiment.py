"""
Case Study III: NL2SQL Explorer-Refiner experiment on WikiSQL.
Compares SingleLLM vs AgenticWorkflow (Explorer-Refiner) on EM and EX.
"""

import io
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent

# ── Config ────────────────────────────────────────────────────────────────────
REFINEMENT_N = 40   # Explorer invokes CDE agent this many times during refinement
TEST_N       = 100  # held-out test queries
RANDOM_SEED  = 7
MODEL        = "gpt-4o-mini"

PARQUET_URL = (
    "https://huggingface.co/datasets/Salesforce/wikisql"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

AGG_OPS  = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', '!=']

# ── Prompts ───────────────────────────────────────────────────────────────────
BASE_SYSTEM = """You are a SQL assistant. Convert natural language questions to SQL queries.
Use the table name and column names provided. Return only the SQL query."""

EXPLORER_SYSTEM = """You are an Explorer agent analyzing failure modes of a NL2SQL system.
You will receive examples where the system generated wrong SQL.
Identify up to 8 concise, actionable failure patterns (e.g. wrong aggregation, wrong column, missing WHERE condition, wrong operator, formatting errors).
Output a JSON list of strings. Example: ["Pattern 1", "Pattern 2"]
Return ONLY the JSON list."""

REFINER_SYSTEM = """You are a Refiner agent. Given a base instruction set for a NL2SQL agent and a list of observed failure patterns from the Explorer, produce an improved instruction set that directly addresses those failures.
Return ONLY the improved system prompt text — no JSON, no preamble."""


# ── Load data ─────────────────────────────────────────────────────────────────
def load_wikisql(n_refinement: int, n_test: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    print("Loading WikiSQL parquet...", flush=True)
    r = requests.get(PARQUET_URL, allow_redirects=True)
    df = pd.read_parquet(io.BytesIO(r.content))
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    examples = []
    for _, row in df.iterrows():
        t   = row["table"]
        s   = row["sql"]
        headers = list(t["header"])
        rows    = [list(r) for r in t["rows"]]
        tname   = "t_" + re.sub(r'[^a-zA-Z0-9_]', '_', t["name"])

        conds = s["conds"]
        examples.append({
            "question":   row["question"],
            "table_name": tname,
            "headers":    headers,
            "types":      list(t["types"]),
            "rows":       rows,
            # structured fields for reference SQL construction
            "sql_sel":    int(s["sel"]),
            "sql_agg":    int(s["agg"]),
            "sql_cond_cols": [int(c) for c in conds["column_index"]],
            "sql_cond_ops":  [int(o) for o in conds["operator_index"]],
            "sql_cond_vals": [str(v) for v in conds["condition"]],
        })
        if len(examples) >= n_refinement + n_test:
            break

    return examples[:n_refinement], examples[n_refinement: n_refinement + n_test]


# ── Build reference SQL from structured fields ────────────────────────────────
def build_ref_sql(ex: Dict) -> str:
    headers = ex["headers"]
    tname   = ex["table_name"]
    agg     = AGG_OPS[ex["sql_agg"]]
    col     = f'"{headers[ex["sql_sel"]]}"'
    select  = f"{agg}({col})" if agg else col

    wheres = []
    for ci, oi, val in zip(ex["sql_cond_cols"], ex["sql_cond_ops"], ex["sql_cond_vals"]):
        op      = COND_OPS[oi] if oi < len(COND_OPS) else "="
        cname   = f'"{headers[ci]}"'
        # numeric type → no quotes; text → single quotes
        typ = ex["types"][ci] if ci < len(ex["types"]) else "text"
        if typ in ("real", "integer") and re.match(r'^-?\d+(\.\d+)?$', str(val)):
            wheres.append(f"{cname} {op} {val}")
        else:
            wheres.append(f"{cname} {op} '{val}'")

    sql = f"SELECT {select} FROM {tname}"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    return sql


# ── SQL execution ─────────────────────────────────────────────────────────────
def _fix_table_ref(sql: str, table_name: str) -> str:
    """Replace bare FROM table / FROM 'table' with actual table name."""
    sql = re.sub(r'\bFROM\s+[\'"]?table[\'"]?\b', f'FROM {table_name}', sql, flags=re.IGNORECASE)
    return sql.strip().rstrip(";")


def execute_sql(sql: str, ex: Dict) -> Optional[List]:
    tname   = ex["table_name"]
    headers = ex["headers"]
    rows    = ex["rows"]
    sql     = _fix_table_ref(sql, tname)
    try:
        con = sqlite3.connect(":memory:")
        safe_hdrs = [f'"{h}"' for h in headers]
        con.execute(f"CREATE TABLE {tname} ({', '.join(safe_hdrs)})")
        con.executemany(f"INSERT INTO {tname} VALUES ({', '.join(['?']*len(headers))})", rows)
        cur = con.execute(sql)
        result = sorted([tuple(str(v) if v is not None else '' for v in r) for r in cur.fetchall()])
        con.close()
        return result
    except Exception:
        return None


# ── LLM ───────────────────────────────────────────────────────────────────────
client = OpenAI()

def llm(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    return resp.choices[0].message.content.strip()


# ── NL2SQL agent ──────────────────────────────────────────────────────────────
def generate_sql(ex: Dict, system: str) -> str:
    schema = (
        f"Table: {ex['table_name']}\n"
        f"Columns: {', '.join(ex['headers'])}\n"
        f"Types:   {', '.join(ex['types'])}"
    )
    user = f"{schema}\n\nQuestion: {ex['question']}\n\nSQL:"
    raw  = llm(system, user)
    raw  = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw  = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────
def exact_match(pred: str, ex: Dict) -> bool:
    ref  = build_ref_sql(ex)
    def norm(s):
        s = _fix_table_ref(s, ex["table_name"]).lower()
        return re.sub(r'\s+', ' ', s).strip()
    return norm(pred) == norm(ref)


def execution_match(pred: str, ex: Dict) -> bool:
    ref_sql  = build_ref_sql(ex)
    r_ref    = execute_sql(ref_sql, ex)
    r_pred   = execute_sql(pred,    ex)
    if r_pred is None or r_ref is None:
        return False
    return r_pred == r_ref


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(examples: List[Dict], system: str, label: str) -> Tuple[float, float, List[Dict]]:
    em_hits = ex_hits = 0
    details = []
    for i, ex in enumerate(examples):
        pred = generate_sql(ex, system)
        em   = exact_match(pred, ex)
        exe  = execution_match(pred, ex)
        em_hits += em
        ex_hits += exe
        details.append({**ex, "pred_sql": pred, "ref_sql": build_ref_sql(ex), "em": em, "ex": exe})
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i+1}/{len(examples)}  EM={em_hits/(i+1):.3f}  EX={ex_hits/(i+1):.3f}", flush=True)
    n = len(examples)
    return em_hits / n, ex_hits / n, details


# ── Explorer ──────────────────────────────────────────────────────────────────
def run_explorer(refinement_set: List[Dict], base_system: str) -> str:
    print(f"  Explorer: running on {len(refinement_set)} examples...", flush=True)
    failures = []
    for ex in refinement_set:
        pred = generate_sql(ex, base_system)
        em   = exact_match(pred, ex)
        exe  = execution_match(pred, ex)
        if not em or not exe:
            failures.append({
                "question": ex["question"],
                "headers":  ex["headers"],
                "ref_sql":  build_ref_sql(ex),
                "pred_sql": pred,
            })
    print(f"  Explorer: {len(failures)}/{len(refinement_set)} failures. Analyzing...", flush=True)
    if not failures:
        return "[]"
    user = "Failed examples:\n" + json.dumps(failures[:15], indent=2)
    return llm(EXPLORER_SYSTEM, user)


# ── Refiner ───────────────────────────────────────────────────────────────────
def run_refiner(base_system: str, observations: str) -> str:
    print("  Refiner: producing refined instruction set...", flush=True)
    user = (
        f"Base instructions:\n{base_system}\n\n"
        f"Failure patterns observed by Explorer:\n{observations}\n\n"
        "Output the improved instruction set:"
    )
    return llm(REFINER_SYSTEM, user)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    refinement_set, test_set = load_wikisql(REFINEMENT_N, TEST_N, RANDOM_SEED)
    print(f"Data: {len(refinement_set)} refinement, {len(test_set)} test.\n")

    # SingleLLM
    print("=== SingleLLM ===")
    t0 = time.time()
    em_s, ex_s, single_details = evaluate(test_set, BASE_SYSTEM, "SingleLLM")
    print(f"  EM={em_s:.4f}  EX={ex_s:.4f}  ({time.time()-t0:.1f}s)\n")

    # AgenticWorkflow
    print("=== AgenticWorkflow ===")
    t0 = time.time()
    observations   = run_explorer(refinement_set, BASE_SYSTEM)
    refined_system = run_refiner(BASE_SYSTEM, observations)
    print(f"\nRefined prompt:\n{refined_system}\n")
    em_a, ex_a, agentic_details = evaluate(test_set, refined_system, "AgenticWF")
    overhead      = REFINEMENT_N + 2  # refinement evals + explorer LLM + refiner LLM
    avg_calls     = 1 + overhead / TEST_N
    print(f"  EM={em_a:.4f}  EX={ex_a:.4f}  avg_calls/query={avg_calls:.2f}  ({time.time()-t0:.1f}s)\n")

    # Cherry-pick worked example: SingleLLM wrong, AgenticWF right
    print("=== Cherry-picked worked example ===")
    for s, a in zip(single_details, agentic_details):
        if not s["ex"] and a["ex"]:
            print(f"Question  : {s['question']}")
            print(f"Headers   : {s['headers']}")
            print(f"Ref SQL   : {s['ref_sql']}")
            print(f"SingleLLM : {s['pred_sql']}")
            print(f"AgenticWF : {a['pred_sql']}")
            break

    print(f"\n=== FINAL RESULTS ===")
    print(f"SingleLLM   EM={em_s*100:.1f}%  EX={ex_s*100:.1f}%  avg_calls=1.0")
    print(f"AgenticWF   EM={em_a*100:.1f}%  EX={ex_a*100:.1f}%  avg_calls={avg_calls:.2f}")
    print(f"Total time  : {time.time()-t_start:.1f}s")

    out_dir = ROOT / "nl2sql"
    (out_dir / "explorer_observations.txt").write_text(observations)
    (out_dir / "refined_system_prompt.txt").write_text(refined_system)
    print("Saved observations and refined prompt.")


if __name__ == "__main__":
    main()
