#!/usr/bin/env python3
"""
Unified data-cleaning utility vs tradeoff weight (responds to reviewer: quality + cost on same axis).

Cost model (linear, all normalized to [0, 1]):
  T̃ = min(1, T_tokens / T_ref),  H̃ = H_human / H_max
  Cost = w_token * T̃ + w_human * H̃

Utility (default matches revised paper text — quality good, cost bad):
  U(α) = α * Quality - (1 - α) * Cost
  with Quality, Cost ∈ [0, 1].

Set SUBTRACT_COST = False for U = α*Q + (1-α)*C (not recommended if Cost is a burden).

Token counts T are order-of-magnitude estimates for N≈2000 rows, chunk=100, gpt-4o-mini-style
prompts (see README section "Unified cleaning utility").

Primary paper-facing outputs (chart + main long CSV) live under:
  final_results/Data_Cleaning_Utility_Quality_Cost/
  - utility_quality_cost_tradeoff.csv — long format (full α grid)
  - utility_quality_cost_tradeoff.png — one line per method

Companion CSVs from the same run:
  - utility_quality_cost_inputs.csv — Q, T, H, normalized Cost per method
  - utility_quality_cost_summary_5alpha.csv — rows at α ∈ {0, 0.25, 0.5, 0.75, 1}

Run from repo root:
  python -m data_cleaning.exp_utility_quality_vs_cost
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "final_results" / "Data_Cleaning_Utility_Quality_Cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference scale for token burden (same units as T); pulls batched cleaning closer to reviewer on T̃.
T_REF = 400_000
H_MAX = 980  # human units for LLM+HITL (paper table)
W_TOKEN = 0.5
W_HUMAN = 0.5

# If True: U = α Q - (1-α) C. If False: U = α Q + (1-α) C
SUBTRACT_COST = True

# Order-of-magnitude total tokens (input+output) for N≈2000, from code structure estimates
METHOD_ROWS: list[dict] = [
    {"method_key": "no_cleaning", "label": "NoCleaning", "quality_frac": 0.0, "tokens": 0, "human_units": 0},
    {"method_key": "rule_based", "label": "RuleBased", "quality_frac": 0.436, "tokens": 0, "human_units": 0},
    {"method_key": "llm_cleaner", "label": "LLM-Cleaner", "quality_frac": 0.948, "tokens": 210_000, "human_units": 0},
    {"method_key": "llm_reviewer_llm", "label": "LLM+ReviewerLLM", "quality_frac": 0.936, "tokens": 1_100_000, "human_units": 0},
    {"method_key": "llm_hitl", "label": "LLM+HITL", "quality_frac": 1.0, "tokens": 280_000, "human_units": 980},
]

N_ALPHA_PLOT = 101
ALPHA_SUMMARY = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def normalized_cost(tokens: float, human: float) -> float:
    t_tilde = min(1.0, float(tokens) / T_REF)
    h_tilde = min(1.0, float(human) / H_MAX) if H_MAX > 0 else 0.0
    return W_TOKEN * t_tilde + W_HUMAN * h_tilde


def utility(alpha: np.ndarray, q: float, c: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if SUBTRACT_COST:
        return a * q - (1.0 - a) * c
    return a * q + (1.0 - a) * c


def main() -> None:
    rows = []
    meta = []
    for m in METHOD_ROWS:
        c = normalized_cost(m["tokens"], m["human_units"])
        meta.append(
            {
                "method_key": m["method_key"],
                "label": m["label"],
                "quality": m["quality_frac"],
                "tokens_est": m["tokens"],
                "human_units": m["human_units"],
                "T_tilde": min(1.0, m["tokens"] / T_REF),
                "H_tilde": min(1.0, m["human_units"] / H_MAX) if H_MAX else 0.0,
                "cost": c,
                "T_ref": T_REF,
                "H_max": H_MAX,
                "w_token": W_TOKEN,
                "w_human": W_HUMAN,
            }
        )

    meta_df = pd.DataFrame(meta)
    meta_path = OUT_DIR / "utility_quality_cost_inputs.csv"
    meta_df.to_csv(meta_path, index=False)

    alphas = np.linspace(0.0, 1.0, N_ALPHA_PLOT)
    for m in METHOD_ROWS:
        q, c = m["quality_frac"], normalized_cost(m["tokens"], m["human_units"])
        u = utility(alphas, q, c)
        for a, val in zip(alphas, u):
            rows.append(
                {
                    "alpha_utility": float(a),
                    "method_key": m["method_key"],
                    "label": m["label"],
                    "quality": q,
                    "cost": c,
                    "utility": float(val),
                }
            )

    long_df = pd.DataFrame(rows)
    long_path = OUT_DIR / "utility_quality_cost_tradeoff.csv"
    long_df.to_csv(long_path, index=False)

    # Summary at 5 α values
    sum_rows = []
    for m in METHOD_ROWS:
        q, c = m["quality_frac"], normalized_cost(m["tokens"], m["human_units"])
        for a in ALPHA_SUMMARY:
            u = float(utility(np.array([a]), q, c)[0])
            sum_rows.append(
                {
                    "alpha_utility": float(a),
                    "method_key": m["method_key"],
                    "label": m["label"],
                    "quality": q,
                    "cost": c,
                    "utility": u,
                }
            )
    pd.DataFrame(sum_rows).to_csv(OUT_DIR / "utility_quality_cost_summary_5alpha.csv", index=False)

    # Plot
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    # Okabe–Ito–style palette + distinct markers + line dashes for color-blind–friendly distinction
    series_style = [
        {"color": "#999999", "marker": "o", "linestyle": "-", "z": 5},
        {"color": "#E69F00", "marker": "^", "linestyle": "--", "z": 5},
        {"color": "#0072B2", "marker": "s", "linestyle": "-", "z": 5},
        {"color": "#CC79A7", "marker": "D", "linestyle": "-.", "z": 5},
        {"color": "#009E73", "marker": "P", "linestyle": ":", "z": 5},
    ]
    markevery = max(1, N_ALPHA_PLOT // 12)
    for i, m in enumerate(METHOD_ROWS):
        sub = long_df[long_df["method_key"] == m["method_key"]]
        st = series_style[i % len(series_style)]
        ax.plot(
            sub["alpha_utility"],
            sub["utility"],
            label=m["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.85,
            marker=st["marker"],
            markersize=5.5,
            markevery=markevery,
            markerfacecolor=st["color"],
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            zorder=st["z"],
        )
    ax.set_xlabel(r"$\alpha$ (quality weight)")
    ax.set_ylabel("Utility(data_cleaning)")
    ax.set_title("Data cleaning: quality vs cost tradeoff")
    ax.axhline(0.0, color="#999", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    png_path = OUT_DIR / "utility_quality_cost_tradeoff.png"
    fig.savefig(png_path, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcdefaults()

    print(f"Wrote {meta_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {OUT_DIR / 'utility_quality_cost_summary_5alpha.csv'}")
    print(f"Wrote {png_path}")
    print(f"  SUBTRACT_COST={SUBTRACT_COST}  T_ref={T_REF}  H_max={H_MAX}  w_token,w_human={W_TOKEN},{W_HUMAN}")


if __name__ == "__main__":
    main()
