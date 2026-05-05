"""
Experiment 2.3: Compositional Monotonicity Heatmap.

Builds a heatmap from existing cross-stage link tables using monotone_support_pct.
No re-estimation is performed.

Row labels match the paper path notation (collection / cleaning / feature / model;
former explore_features shown as feature; no Q_ prefixes). Hats indicate the
propagation source stage(s) per link definition.

Run:
  python -m utility_propagation.exp_compositional_monotonicity_heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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

# Matplotlib mathtext paths aligned with the paper link table (same order as cross-stage rows).
_KEY = Tuple[str, str]
_DISPLAY_MATHTEXT: dict[_KEY, str] = {
    ("Q_collection", "Q_cleaning"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \mathrm{cleaning}$"
    ),
    ("Q_collection", "Q_explore_features"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \mathrm{cleaning} \rightarrow \mathrm{feature}$"
    ),
    ("Q_collection", "Q_model_cv"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \mathrm{cleaning} \rightarrow "
        r"\mathrm{feature} \rightarrow \mathrm{model}$"
    ),
    ("Q_cleaning", "Q_explore_features"): (
        r"$\widehat{\mathrm{cleaning}} \rightarrow \mathrm{feature}$"
    ),
    ("Q_cleaning", "Q_model_cv"): (
        r"$\widehat{\mathrm{cleaning}} \rightarrow \mathrm{feature} \rightarrow \mathrm{model}$"
    ),
    ("Q_explore_features", "Q_model_cv"): (
        r"$\widehat{\mathrm{feature}} \rightarrow \mathrm{model}$"
    ),
    ("Q_collection + Q_cleaning", "Q_explore_features"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \widehat{\mathrm{cleaning}} \rightarrow "
        r"\mathrm{feature}$"
    ),
    ("Q_collection + Q_explore_features", "Q_model_cv"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \mathrm{cleaning} \rightarrow "
        r"\widehat{\mathrm{feature}} \rightarrow \mathrm{model}$"
    ),
    ("Q_collection + Q_cleaning + Q_explore_features", "Q_model_cv"): (
        r"$\widehat{\mathrm{collection}} \rightarrow \widehat{\mathrm{cleaning}} \rightarrow "
        r"\widehat{\mathrm{feature}} \rightarrow \mathrm{model}$"
    ),
    ("Q_cleaning + Q_explore_features", "Q_model_cv"): (
        r"$\widehat{\mathrm{cleaning}} \rightarrow \widehat{\mathrm{feature}} \rightarrow "
        r"\mathrm{model}$"
    ),
}


def _display_label(source_stage: str, target_stage: str) -> str:
    key = (source_stage.strip(), target_stage.strip())
    if key not in _DISPLAY_MATHTEXT:
        raise KeyError(f"No display label for link {key}")
    return _DISPLAY_MATHTEXT[key]


def _load(dataset: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected table: {path}")
    df = pd.read_csv(path)
    need = {"source_stage", "target_stage", "monotone_support_pct"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path.name} is missing required columns: {sorted(need)}")
    out = df.loc[:, ["source_stage", "target_stage", "monotone_support_pct"]].copy()
    out["dataset"] = dataset
    out["link_display"] = [
        _display_label(s, t) for s, t in zip(out["source_stage"], out["target_stage"])
    ]
    return out.loc[:, ["dataset", "link_display", "monotone_support_pct"]]


def _build_matrix() -> pd.DataFrame:
    adult = _load("adult", ADULT_TABLE)
    bank = _load("bank", BANK_TABLE)
    both = pd.concat([adult, bank], ignore_index=True)

    link_order = (
        both.groupby("link_display", as_index=False)["monotone_support_pct"]
        .mean()
        .sort_values("monotone_support_pct", ascending=False)["link_display"]
        .tolist()
    )
    mat = both.pivot(index="link_display", columns="dataset", values="monotone_support_pct")
    mat = mat.reindex(index=link_order, columns=["adult", "bank"])
    mat = mat.rename(columns={"adult": "Adult_Income", "bank": "Bank_Marketing"})
    return mat


def _plot(mat: pd.DataFrame) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    data = mat.to_numpy(dtype=float)
    fig_h = max(5.0, 0.52 * len(mat.index))
    fig_w = 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_title("Cross-Stage Utility Propagation Support", fontsize=11.5, pad=12)
    ax.set_xlabel("Dataset", fontsize=10, labelpad=8)
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist())
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=7.2)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "white" if val >= 65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=7.8)

    # Colorbar: keep pad small so the axis and bar stay visually grouped (pad is fraction of parent Axes width).
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Monotone Support (%)", fontsize=10, labelpad=8)

    fig.subplots_adjust(left=0.30, right=0.92, top=0.93, bottom=0.14)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    mat = _build_matrix()
    mat.to_csv(OUT_CSV)
    _plot(mat)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
