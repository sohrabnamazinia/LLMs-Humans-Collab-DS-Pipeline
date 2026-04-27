"""
Experiment 2.1: cross-stage propagation tables (per dataset).

Run:
  python -m utility_propagation.exp_cross_stage_tables --dataset both

Outputs:
  utility_propagation/outputs/TABLE_cross_stage_link_effects_<dataset>_<stamp>.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_GRID_PREFIX = "propagation_grid_results_"
_STAGE_COLS = ("Q_collection", "Q_cleaning", "Q_explore_features", "Q_model_cv")
_LINKS = (
    (("Q_collection",), "Q_cleaning"),
    (("Q_collection",), "Q_explore_features"),
    (("Q_collection",), "Q_model_cv"),
    (("Q_cleaning",), "Q_explore_features"),
    (("Q_cleaning",), "Q_model_cv"),
    (("Q_explore_features",), "Q_model_cv"),
    # requested multi-source cross-stage links
    (("Q_collection", "Q_cleaning"), "Q_explore_features"),
    (("Q_collection", "Q_explore_features"), "Q_model_cv"),
    (("Q_collection", "Q_cleaning", "Q_explore_features"), "Q_model_cv"),
    (("Q_cleaning", "Q_explore_features"), "Q_model_cv"),
)
_TARGET_BASE_EPS = 1e-6
_RATIO_CLIP_ABS = 10.0  # +/-1000%
_N_BINS = 6


def _legacy_adult_grid_name(name: str) -> bool:
    return bool(re.fullmatch(r"propagation_grid_results_\d{8}_\d{6}\.csv", name))


def _pick_grid_csv(dataset: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_grids = list(OUTPUT_DIR.glob(f"{_GRID_PREFIX}*.csv"))
    if dataset == "adult":
        cand = [
            p
            for p in all_grids
            if p.name.startswith(f"{_GRID_PREFIX}adult_") or _legacy_adult_grid_name(p.name)
        ]
    else:
        cand = [p for p in all_grids if p.name.startswith(f"{_GRID_PREFIX}{dataset}_")]
    if not cand:
        raise FileNotFoundError(f"No grid CSV found for dataset={dataset!r} in {OUTPUT_DIR}")
    return max(cand, key=lambda p: p.stat().st_mtime).resolve()


def _stamp_from_grid(path: Path) -> str:
    m = re.match(r"propagation_grid_results_(.+)\.csv", path.name)
    return m.group(1) if m else "latest"


def _binned_pairwise_target_deltas_and_ratio_pct(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[ok], dtype=float)
    y = np.asarray(y[ok], dtype=float)
    if len(x) < 3:
        return np.array([], dtype=float), np.array([], dtype=float), int(len(x))

    x_s = pd.Series(x)
    n_bins = max(3, int(n_bins))
    bins = pd.qcut(x_s, q=n_bins, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "b": bins}).groupby("b", observed=True).agg(
        x_mean=("x", "mean"), y_mean=("y", "mean")
    )
    if len(g) < 2:
        return np.array([], dtype=float), np.array([], dtype=float), int(len(x))
    g = g.sort_values("x_mean")
    xb = g["x_mean"].to_numpy(dtype=float)
    yb = g["y_mean"].to_numpy(dtype=float)

    xs = xb[:, None] - xb[None, :]
    ys = yb[:, None] - yb[None, :]
    mask = xs < -1e-12
    dy = ys[mask].astype(float)
    y_before = np.broadcast_to(yb[:, None], xs.shape)[mask].astype(float)
    denom = np.maximum(np.abs(y_before), _TARGET_BASE_EPS)
    ratio = dy / denom
    ratio = np.clip(ratio, -_RATIO_CLIP_ABS, _RATIO_CLIP_ABS)
    ratio_pct = 100.0 * ratio
    return dy, ratio_pct, int(len(x))


def _local_link_stats(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, n_bins: int) -> dict:
    dy, ratio_pct, n_obs = _binned_pairwise_target_deltas_and_ratio_pct(x, y, n_bins=n_bins)
    n_pairs = int(len(ratio_pct))
    if n_pairs == 0:
        return {
            "cross_stage_utility_propagation_effect_pct": np.nan,
            "p_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "violation_rate_pct": np.nan,
            "monotone_support_pct": np.nan,
            "n_pairs": 0,
            "n_obs": n_obs,
        }

    effect = float(np.mean(ratio_pct))
    violation = 100.0 * float(np.mean(dy < 0.0))
    support = 100.0 - violation

    rs = np.random.default_rng(seed)
    boot = []
    idx = np.arange(n_pairs)
    for _ in range(max(200, int(n_boot))):
        b = rs.choice(idx, size=n_pairs, replace=True)
        boot.append(float(np.mean(ratio_pct[b])))
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    ge0 = float(np.mean(np.asarray(boot) >= 0.0))
    le0 = float(np.mean(np.asarray(boot) <= 0.0))
    pval = 2.0 * min(ge0, le0)
    pval = max(0.0, min(1.0, pval))

    return {
        "cross_stage_utility_propagation_effect_pct": effect,
        "p_value": pval,
        "ci95_low": lo,
        "ci95_high": hi,
        "violation_rate_pct": violation,
        "monotone_support_pct": support,
        "n_pairs": n_pairs,
        "n_obs": n_obs,
    }


def _link_rows(df: pd.DataFrame, dataset: str, n_boot: int, seed: int, n_bins: int) -> list[dict]:
    rows: list[dict] = []
    for i, (src_cols, tgt) in enumerate(_LINKS):
        x = df.loc[:, list(src_cols)].mean(axis=1).to_numpy(dtype=float)
        y = df[tgt].to_numpy(dtype=float)
        est = _local_link_stats(x, y, n_boot=n_boot, seed=seed + 1009 * (i + 1), n_bins=n_bins)
        eff = est["cross_stage_utility_propagation_effect_pct"]
        direction = "positive" if np.isfinite(eff) and eff > 0 else "negative" if np.isfinite(eff) and eff < 0 else "neutral"
        src_name = " + ".join(src_cols)
        rows.append(
            {
                "dataset": dataset,
                "source_stage": src_name,
                "target_stage": tgt,
                "n_pairs": est["n_pairs"],
                "direction": direction,
                "cross_stage_utility_propagation_effect_pct": eff,
                "ci95_low_pct": est["ci95_low"],
                "ci95_high_pct": est["ci95_high"],
                "violation_rate_pct": est["violation_rate_pct"],
                "monotone_support_pct": est["monotone_support_pct"],
                "p_value": est["p_value"],
                "n_obs": est["n_obs"],
            }
        )
    return rows


def _run_dataset(dataset: str, n_boot: int, seed: int, n_bins: int) -> Path:
    grid = _pick_grid_csv(dataset)
    df = pd.read_csv(grid)
    keep = [c for c in _STAGE_COLS if c in df.columns]
    if len(keep) != 4:
        raise ValueError(f"Missing required stage columns in {grid.name}: {_STAGE_COLS}")
    rows = _link_rows(df, dataset=dataset, n_boot=n_boot, seed=seed, n_bins=n_bins)
    out = pd.DataFrame(rows)
    stamp = _stamp_from_grid(grid)
    pref = f"{dataset}_"
    if stamp.startswith(pref):
        stamp = stamp[len(pref) :]
    path = OUTPUT_DIR / f"TABLE_cross_stage_link_effects_{dataset}_{stamp}.csv"
    out.to_csv(path, index=False)
    return path


def main(dataset: str, n_boot: int, seed: int, n_bins: int) -> None:
    ds: Iterable[str] = ("adult", "bank") if dataset == "both" else (dataset,)
    for d in ds:
        out = _run_dataset(d, n_boot=n_boot, seed=seed, n_bins=n_bins)
        print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 2.1 cross-stage link table(s)")
    ap.add_argument("--dataset", choices=("adult", "bank", "both"), default="both")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins", type=int, default=_N_BINS, help="Quantile bins for source-stage discretization")
    args = ap.parse_args()
    main(dataset=args.dataset, n_boot=args.n_boot, seed=args.seed, n_bins=args.n_bins)
