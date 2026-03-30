# -----------------------------------------------------------------------------
# Run from repository root:
#   python3 -m utility_propagation.run_grid
#   python3 -m utility_propagation.fit_propagation
#
# Optional — pick a specific grid CSV (must live under utility_propagation/outputs/):
#   python3 -m utility_propagation.fit_propagation -i propagation_grid_results_<timestamp>.csv
# -----------------------------------------------------------------------------
"""
Fit test_accuracy ~ degree-2 polynomial in standardized stage-Q metrics.

Outputs: formula comparison CSV, coefficient tables (raw + display-scaled), MSE bar chart (×10³),
sample configs CSV, feature×model heatmap.

Fits: scipy.optimize.lsq_linear with nonnegative coefficients on the four linear Q terms; squares and
interactions unconstrained.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_GRID_PREFIX = "propagation_grid_results_"

FEATURE_COLS = [
    "Q_collection",
    "Q_cleaning",
    "Q_explore_features",
    "Q_model_cv",
]

# Fixed total degree for propagation surface
POLY_DEGREE = 2

# First four polynomial columns = linear effects of the four pipeline stages (standardized Q).
_N_STAGE_LINEAR_TERMS = 4

# One cross-term dropped in the “near-full” ablation (must match sklearn feature name)
_DROPPED_CROSS_TERM = "Q_explore_features Q_model_cv"

_MSE_PLOT_SCALE = 1e3
_SLOPE_DISPLAY_SCALE = 100.0
_INTERCEPT_DISPLAY_SCALE = 100.0


def _pick_grid_csv(explicit: Optional[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = OUTPUT_DIR / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    stamped = list(OUTPUT_DIR.glob(f"{_GRID_PREFIX}*.csv"))
    if stamped:
        return max(stamped, key=lambda x: x.stat().st_mtime).resolve()
    legacy = OUTPUT_DIR / "propagation_grid_results.csv"
    if legacy.is_file():
        return legacy.resolve()
    raise FileNotFoundError(
        f"No grid CSV in {OUTPUT_DIR}. Run python -m utility_propagation.run_grid first."
    )


def _timestamp_from_grid_path(path: Path) -> str:
    if path.name.startswith(_GRID_PREFIX) and path.suffix == ".csv":
        return path.name[len(_GRID_PREFIX) : -4]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _paper_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _degree2_design(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler, PolynomialFeatures]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=False, include_bias=False)
    Z = poly.fit_transform(Xs)
    return Z, scaler, poly


def _term_masks(feature_names: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
    """Boolean column masks for five nested / ablated quadratic designs (same column order as sklearn)."""
    names = list(feature_names)
    is_lin = np.array([(("^" not in n) and (" " not in n)) for n in names], dtype=bool)
    is_sq = np.array([("^2" in n) for n in names], dtype=bool)
    is_cross = np.array([(" " in n) and ("^2" not in n) for n in names], dtype=bool)

    mask_all = np.ones(len(names), dtype=bool)
    mask_minus = mask_all.copy()
    if _DROPPED_CROSS_TERM in names:
        mask_minus[names.index(_DROPPED_CROSS_TERM)] = False
    else:
        raise ValueError(f"Expected term {_DROPPED_CROSS_TERM!r} in polynomial names {names}")

    return [
        ("linear", "Linear", is_lin),
        ("main_squares", "Main + squares", is_lin | is_sq),
        ("main_crosses", "Main + crosses", is_lin | is_cross),
        ("minus_one_cross", f"Full − ({_DROPPED_CROSS_TERM.replace(' ', '×')})", mask_minus),
        ("full_quadratic", "Full quadratic", mask_all),
    ]


def _constrained_stage_monotone_fit(
    Z: np.ndarray, y: np.ndarray, mask: np.ndarray, n_poly: int
) -> Tuple[float, np.ndarray, float, float]:
    """Bounded LSQ: linear stage columns (indices < 4) ≥ 0 when active. Returns intercept, coef, MSE, R²."""
    cols = np.flatnonzero(mask)
    full = np.zeros(n_poly, dtype=float)
    if cols.size == 0:
        return 0.0, full, float("inf"), float("nan")

    Xs = Z[:, cols]
    n, p = Xs.shape
    A = np.column_stack([np.ones(n, dtype=float), Xs])
    lb = np.full(1 + p, -np.inf, dtype=float)
    ub = np.full(1 + p, np.inf, dtype=float)
    for j, cidx in enumerate(cols):
        if cidx < _N_STAGE_LINEAR_TERMS:
            lb[1 + j] = 0.0

    res = lsq_linear(A, y, bounds=(lb, ub), method="trf", max_iter=3000, verbose=0)
    x = res.x
    if not res.success or not np.all(np.isfinite(x)):
        lr = LinearRegression().fit(Xs, y)
        full[cols] = lr.coef_
        pred = lr.predict(Xs)
        return (
            float(lr.intercept_),
            full,
            float(mean_squared_error(y, pred)),
            float(r2_score(y, pred)),
        )

    full[cols] = x[1:]
    pred = A @ x
    return (
        float(x[0]),
        full,
        float(mean_squared_error(y, pred)),
        float(r2_score(y, pred)),
    )


def _assert_stage_main_effects_nonneg(w: np.ndarray, mask: np.ndarray, tol: float = 1e-8) -> None:
    for k in range(min(_N_STAGE_LINEAR_TERMS, len(w))):
        if mask[k] and w[k] < -tol:
            raise RuntimeError(f"Linear coef {k} violated bound (got {w[k]}).")


def _build_coefficient_table_f1_to_f5(
    poly_names: np.ndarray,
    coef_per_spec: List[np.ndarray],
    intercept_per_spec: List[float],
) -> pd.DataFrame:
    """Wide table: term × f1..f5 from precomputed constrained fits."""
    n_poly = len(poly_names)
    if len(coef_per_spec) != 5 or len(intercept_per_spec) != 5:
        raise ValueError("Expected five specifications.")

    rows = []
    rows.append({"term": "intercept", **{f"f{i}": intercept_per_spec[i - 1] for i in range(1, 6)}})
    for j in range(n_poly):
        rows.append(
            {
                "term": str(poly_names[j]),
                **{f"f{i}": float(coef_per_spec[i - 1][j]) for i in range(1, 6)},
            }
        )

    return pd.DataFrame(rows)


def _table_display_transform(table_raw: pd.DataFrame, y_mean: float) -> pd.DataFrame:
    """Scale polynomial columns × _SLOPE_DISPLAY_SCALE; intercept row → (β₀ − y_mean) × _INTERCEPT_DISPLAY_SCALE."""
    out = table_raw.copy()
    fn_cols = [f"f{i}" for i in range(1, 6)]
    is_intercept = out["term"].eq("intercept")
    out.loc[is_intercept, fn_cols] = (out.loc[is_intercept, fn_cols] - y_mean) * _INTERCEPT_DISPLAY_SCALE
    out.loc[~is_intercept, fn_cols] = out.loc[~is_intercept, fn_cols] * _SLOPE_DISPLAY_SCALE
    return out


def _write_sample_configs(df: pd.DataFrame, path: Path, n: int = 15, seed: int = 42) -> None:
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=seed)
    cols = [
        "n_rows",
        "cleaner",
        "feature_group",
        "model_preset",
        *FEATURE_COLS,
        "test_accuracy",
    ]
    cols = [c for c in cols if c in sample.columns]
    sample[cols].to_csv(path, index=False)


def _plot_heatmap(df: pd.DataFrame, path: Path) -> None:
    _paper_rc()
    piv = df.groupby(["feature_group", "model_preset"], observed=True)["test_accuracy"].mean().unstack()
    piv = piv.reindex(sorted(piv.index))
    piv = piv.reindex(columns=sorted(piv.columns))

    vals = piv.values.astype(float)
    fig_w, fig_h = 5.4, 3.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=vals.min(), vmax=vals.max())
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels(list(piv.columns), rotation=0)
    ax.set_yticklabels(list(piv.index))
    ax.set_xlabel("Model preset (stage 4)")
    ax.set_ylabel("Feature group (stage 3)")
    ax.set_title("Mean test accuracy: feature group × model preset\n(marginal over cleaners × sample sizes)", fontsize=10)

    thr = (vals.min() + vals.max()) / 2.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                color="white" if v > thr else "#1b263b",
                fontsize=8.5,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean test accuracy", fontsize=9)

    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcdefaults()


def _plot_formula_mse_bar(
    short_labels: Sequence[str],
    train_mse: List[float],
    path: Path,
    mse_plot_scale: float = _MSE_PLOT_SCALE,
) -> None:
    _paper_rc()
    x = np.arange(len(short_labels))
    mse = [max(m * mse_plot_scale, 1e-16) for m in train_mse]
    arr = np.asarray(mse, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    span = mx - mn if mx > mn else max(mx * 0.01, 1e-9)
    baseline = max(mn - 0.45 * span, 0.0)
    heights = arr - baseline
    colors = ["#3d5a80"] * (len(mse) - 1) + ["#2a6f4a"]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.bar(
        x,
        heights,
        bottom=baseline,
        color=colors,
        edgecolor="#1b263b",
        linewidth=0.75,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, ha="center", fontsize=9, rotation=0)
    ax.set_ylabel(r"prediction MSE ($10^3$)")
    ax.set_xlabel("Utility formula")
    ax.set_title(
        "Comparing representative utility formulas for pipeline utility prediction",
        fontsize=10.5,
        pad=8,
    )
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(baseline, mx + 0.12 * span)

    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcdefaults()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=None, help="Grid CSV path (default: newest propagation_grid_results_*.csv)")
    args = ap.parse_args()
    try:
        input_csv = _pick_grid_csv(args.input)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    stamp = _timestamp_from_grid_path(input_csv)
    out_table_weights = OUTPUT_DIR / f"TABLE_propagation_polynomial_weights_{stamp}.csv"
    out_table_raw = OUTPUT_DIR / f"TABLE_propagation_polynomial_weights_{stamp}_raw_coefficients.csv"
    out_compare = OUTPUT_DIR / f"propagation_formula_comparison_{stamp}.csv"
    out_report = OUTPUT_DIR / f"propagation_fit_report_{stamp}.txt"
    out_sample = OUTPUT_DIR / f"propagation_sample_configs_{stamp}.csv"
    out_bar = OUTPUT_DIR / f"propagation_chart_formula_train_mse_{stamp}.png"
    out_heat = OUTPUT_DIR / f"propagation_chart_feature_model_heatmap_{stamp}.png"

    print(f"Loading grid results from {input_csv} ...")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["test_accuracy"])
    if len(df) < 10:
        print("Not enough valid rows after dropping NaN metrics.")
        sys.exit(1)

    df = df.copy()
    df["Q_model_cv"] = df["Q_model_cv"].fillna(df["Q_model_cv"].median())
    X = df[FEATURE_COLS].values
    y = df["test_accuracy"].values

    Z, _, poly_fitted = _degree2_design(X)
    names = poly_fitted.get_feature_names_out(FEATURE_COLS)
    specs = _term_masks(names)

    n_poly = len(names)
    compare_rows = []
    mse_list: List[float] = []
    intercepts: List[float] = []
    coefs: List[np.ndarray] = []

    for key, label, mask in specs:
        b, w, mse_tr, r2_tr = _constrained_stage_monotone_fit(Z, y, mask, n_poly)
        _assert_stage_main_effects_nonneg(w, mask)
        n_terms = int(mask.sum())
        compare_rows.append(
            {
                "model_key": key,
                "label": label,
                "n_terms": n_terms,
                "train_mse_constrained_lsq": mse_tr,
                "train_r2_constrained_lsq": r2_tr,
                "dropped_term": _DROPPED_CROSS_TERM if key == "minus_one_cross" else "",
            }
        )
        mse_list.append(mse_tr)
        intercepts.append(b)
        coefs.append(w)

    bar_labels = [f"f{i}" for i in range(1, 6)]

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_compare, index=False)

    table_df = _build_coefficient_table_f1_to_f5(names, coefs, intercepts)
    table_df.to_csv(out_table_raw, index=False)
    y_mean = float(np.mean(y))
    _table_display_transform(table_df, y_mean).to_csv(out_table_weights, index=False)

    full_mse = float(compare_df.loc[compare_df["model_key"] == "full_quadratic", "train_mse_constrained_lsq"].iloc[0])
    full_r2 = float(compare_df.loc[compare_df["model_key"] == "full_quadratic", "train_r2_constrained_lsq"].iloc[0])

    lines = [
        "Outcome: test_accuracy ~ degree-2 polynomial in standardized Q_collection, Q_cleaning, Q_explore_features, Q_model_cv",
        "Fit: scipy.optimize.lsq_linear; four linear Q coefficients constrained nonnegative when the term is in the model.",
        f"f5 full quadratic: train MSE={full_mse:.6e}, R²={full_r2:.4f}",
        "",
        "Formula comparison:",
        compare_df.to_string(index=False),
        "",
        "TABLE_propagation_polynomial_weights (display):",
        f"  Polynomial rows × {_SLOPE_DISPLAY_SCALE:g}; intercept row (β₀ − mean y) × {_INTERCEPT_DISPLAY_SCALE:g}.",
        f"  Raw coefficients: {out_table_raw.name}",
        "  f1 linear | f2 mains+squares | f3 mains+crosses | f4 full minus one cross | f5 full quadratic",
        f"  {out_table_weights.name}",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    _plot_formula_mse_bar(bar_labels, mse_list, out_bar)
    _write_sample_configs(df, out_sample)
    _plot_heatmap(df, out_heat)

    print(f"\nWrote {out_compare}")
    print(f"Wrote {out_table_raw}")
    print(f"Wrote {out_table_weights}")
    print(f"Wrote {out_report}")
    print(f"Wrote {out_sample}")
    print(f"Wrote {out_bar}")
    print(f"Wrote {out_heat}")


if __name__ == "__main__":
    main()
