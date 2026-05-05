"""
Plot representative alpha-utility pipelines (color + marker shape).

Run:
  python -m utility_propagation.plot_representative_alpha_pipelines

Outputs:
  utility_propagation/outputs/TABLE_alpha_pipeline_curves_representative5_<stamp>.csv
  utility_propagation/outputs/FIG_alpha_pipeline_curves_representative5_<stamp>.png
  utility_propagation/outputs/TABLE_alpha_pipeline_curves_representative5_includes_naive_<stamp>.csv
  utility_propagation/outputs/FIG_alpha_pipeline_curves_representative5_includes_naive_<stamp>.png
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "utility_propagation" / "outputs"

REPRESENTATIVE_PRIMARY_PICK = [
    "P02_Rawnocleaning_SingleLLM",
    "P05_Rulebased_SingleLLM",
    "P08_LLMonly_SingleLLM",
    "P11_LLMplushumanfewshot_SingleLLM",
    "P12_LLMplushumanfewshot_AgenticWorkflow",
]

# Alternate set: include one naive-baseline pipeline for comparison.
REPRESENTATIVE_PICK_INCLUDES_NAIVE = [
    "P04_Rulebased_NaiveBaseline",
    "P05_Rulebased_SingleLLM",
    "P08_LLMonly_SingleLLM",
    "P11_LLMplushumanfewshot_SingleLLM",
    "P12_LLMplushumanfewshot_AgenticWorkflow",
]

STYLE = {
    "P02_Rawnocleaning_SingleLLM": {"color": "#1f77b4", "marker": "o"},
    "P04_Rulebased_NaiveBaseline": {"color": "#1f77b4", "marker": "P"},
    "P05_Rulebased_SingleLLM": {"color": "#2ca02c", "marker": "s"},
    "P08_LLMonly_SingleLLM": {"color": "#ff7f0e", "marker": "^"},
    "P11_LLMplushumanfewshot_SingleLLM": {"color": "#9467bd", "marker": "D"},
    "P12_LLMplushumanfewshot_AgenticWorkflow": {"color": "#d62728", "marker": "X"},
}


def _latest_curves_csv() -> Path:
    cands = [
        p
        for p in OUT_DIR.glob("TABLE_alpha_pipeline_curves_*.csv")
        if "cherrypicked" not in p.name.lower() and "representative5" not in p.name
    ]
    cands = sorted(cands, key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(
            f"No full alpha-pipeline curves CSV in {OUT_DIR} "
            f"(expected TABLE_alpha_pipeline_curves_<stamp>.csv from exp_alpha_pipeline_representatives)"
        )
    return cands[-1]


def _plot_one_set(df: pd.DataFrame, pick: list[str], out_csv: Path, out_png: Path, title: str) -> None:
    picked = df[df["pipeline_id"].isin(pick)].copy()
    if picked.empty:
        raise ValueError("No selected pipeline IDs found in source curve table.")
    picked.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for pid in pick:
        g = picked[picked["pipeline_id"] == pid].sort_values("alpha")
        if g.empty:
            continue
        st = STYLE.get(pid, {"color": None, "marker": "o"})
        label = f"{pid} ({g['cleaning_method'].iloc[0]} | {g['model_method'].iloc[0]})"
        ax.plot(
            g["alpha"],
            g["utility"],
            label=label,
            color=st["color"],
            marker=st["marker"],
            linewidth=1.8,
            markersize=4.8,
            markevery=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Alpha")
    ax.set_ylabel(r"Utility$_{\alpha}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=7.3, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = _latest_curves_csv()
    df = pd.read_csv(src)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv_primary = OUT_DIR / f"TABLE_alpha_pipeline_curves_representative5_{stamp}.csv"
    out_png_primary = OUT_DIR / f"FIG_alpha_pipeline_curves_representative5_{stamp}.png"
    _plot_one_set(
        df=df,
        pick=REPRESENTATIVE_PRIMARY_PICK,
        out_csv=out_csv_primary,
        out_png=out_png_primary,
        title="Comparison of Pipelines Utility",
    )

    out_csv_secondary = OUT_DIR / f"TABLE_alpha_pipeline_curves_representative5_includes_naive_{stamp}.csv"
    out_png_secondary = OUT_DIR / f"FIG_alpha_pipeline_curves_representative5_includes_naive_{stamp}.png"
    _plot_one_set(
        df=df,
        pick=REPRESENTATIVE_PICK_INCLUDES_NAIVE,
        out_csv=out_csv_secondary,
        out_png=out_png_secondary,
        title="Comparison of Pipelines Utility",
    )

    print(f"Source: {src}")
    print(f"Wrote {out_csv_primary}")
    print(f"Wrote {out_png_primary}")
    print(f"Wrote {out_csv_secondary}")
    print(f"Wrote {out_png_secondary}")


if __name__ == "__main__":
    main()
