"""
Experiment 2.3: Compositional Monotonicity Heatmap.

Builds a heatmap from existing cross-stage link tables using monotone_support_pct.
No re-estimation is performed.

Run:
  python -m utility_propagation.exp_compositional_monotonicity_heatmap
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "utility_propagation" / "outputs"
FINAL_DIR = ROOT / "final_results" / "cross_stage_utility_propagation"

ADULT_TABLE = INPUT_DIR / "TABLE_cross_stage_link_effects_adult_20260329_163213.csv"
BANK_TABLE = INPUT_DIR / "TABLE_cross_stage_link_effects_bank_20260424_192526.csv"

OUT_PNG = FINAL_DIR / "FIG_cross_stage_utility_propagation_compositional_monotonicity_support.png"
OUT_CSV = FINAL_DIR / "TABLE_cross_stage_utility_propagation_compositional_monotonicity_support_matrix.csv"


def _load(dataset: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected table: {path}")
    df = pd.read_csv(path)
    need = {"source_stage", "target_stage", "monotone_support_pct"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path.name} is missing required columns: {sorted(need)}")
    out = df.loc[:, ["source_stage", "target_stage", "monotone_support_pct"]].copy()
    out["dataset"] = dataset
    out["link"] = out["source_stage"] + " -> " + out["target_stage"]
    return out.loc[:, ["dataset", "link", "monotone_support_pct"]]


def _build_matrix() -> pd.DataFrame:
    adult = _load("adult", ADULT_TABLE)
    bank = _load("bank", BANK_TABLE)
    both = pd.concat([adult, bank], ignore_index=True)

    link_order = (
        both.groupby("link", as_index=False)["monotone_support_pct"]
        .mean()
        .sort_values("monotone_support_pct", ascending=False)["link"]
        .tolist()
    )
    mat = both.pivot(index="link", columns="dataset", values="monotone_support_pct")
    mat = mat.reindex(index=link_order, columns=["adult", "bank"])
    mat = mat.rename(columns={"adult": "Adult_Income", "bank": "Bank_Marketing"})
    return mat


def _plot(mat: pd.DataFrame) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    data = mat.to_numpy(dtype=float)
    fig_h = max(5.0, 0.45 * len(mat.index))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_title("Cross-Stage Utility Propagation Support", fontsize=11.5, pad=9)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist())
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "white" if val >= 65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=7.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Monotone Support (%)")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)


def main() -> None:
    mat = _build_matrix()
    mat.to_csv(OUT_CSV)
    _plot(mat)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

