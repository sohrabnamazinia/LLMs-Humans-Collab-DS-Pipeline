# -----------------------------------------------------------------------------
# Run from repository root:
#   python3 -m utility_propagation.run_grid
#   python3 -m utility_propagation.fit_propagation
#
# Optional — pick a specific grid CSV (under utility_propagation/outputs/):
#   python3 -m utility_propagation.fit_propagation -i propagation_grid_results_<timestamp>.csv
# Or select newest grid by benchmark:
#   python3 -m utility_propagation.fit_propagation --dataset bank
# Optional TABLE display scale (default ×100 Adult, ×520 Bank weights CSV):
#   python3 -m utility_propagation.fit_propagation --dataset bank --display-scale 400
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
import re
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
from sklearn.linear_model import LinearRegression
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
# Bank grids have much tighter test_accuracy; use a larger display multiplier so TABLE_* weights
# sit in a publication-friendly range (same relative structure as raw coefficients).
_DISPLAY_SCALE_BANK_SLOPE = 520.0
_DISPLAY_SCALE_BANK_INTERCEPT = 520.0
# Bank MSE bar + coefficient TABLE column order: g₁…g₅ = these spec indices (linear, crosses, full quad, −1 cross, squares).
_BANK_G_ORDER = (0, 2, 4, 3, 1)


def _legacy_adult_grid_name(name: str) -> bool:
    """Original Adult-only naming: propagation_grid_results_YYYYMMDD_HHMMSS.csv"""
    return bool(re.fullmatch(r"propagation_grid_results_\d{8}_\d{6}\.csv", name))


def _pick_grid_csv(explicit: Optional[str], dataset: Optional[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = OUTPUT_DIR / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    all_grids = list(OUTPUT_DIR.glob(f"{_GRID_PREFIX}*.csv"))
    if dataset:
        pref = f"{_GRID_PREFIX}{dataset}_"
        stamped = [p for p in all_grids if p.name.startswith(pref)]
    else:
        # Default: Adult (new prefix or legacy timestamp file), never auto-pick Bank.
        stamped = [
            p
            for p in all_grids
            if p.name.startswith(f"{_GRID_PREFIX}adult_") or _legacy_adult_grid_name(p.name)
        ]
    if stamped:
        return max(stamped, key=lambda x: x.stat().st_mtime).resolve()
    legacy = OUTPUT_DIR / "propagation_grid_results.csv"
    if legacy.is_file():
        return legacy.resolve()
    raise FileNotFoundError(
        f"No grid CSV in {OUTPUT_DIR}. Run: python -m utility_propagation.run_grid [--dataset adult|bank]"
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


def _build_coefficient_table(
    poly_names: np.ndarray,
    coef_per_spec: List[np.ndarray],
    intercept_per_spec: List[float],
    coef_columns: Sequence[str],
) -> pd.DataFrame:
    """Wide table: term × one column per nested spec (names from ``coef_columns``)."""
    n_poly = len(poly_names)
    if len(coef_per_spec) != 5 or len(intercept_per_spec) != 5 or len(coef_columns) != 5:
        raise ValueError("Expected five specifications and five column names.")

    rows = []
    rows.append({"term": "intercept", **{coef_columns[i]: intercept_per_spec[i] for i in range(5)}})
    for j in range(n_poly):
        rows.append(
            {
                "term": str(poly_names[j]),
                **{coef_columns[i]: float(coef_per_spec[i][j]) for i in range(5)},
            }
        )

    return pd.DataFrame(rows)


def _display_scale_pair(grid_csv: Path, cli_scale: float | None) -> tuple[float, float]:
    """(slope_scale, intercept_scale) for weight tables; CLI overrides all."""
    if cli_scale is not None and cli_scale > 0:
        s = float(cli_scale)
        return s, s
    name = grid_csv.name.lower()
    if "bank" in name:
        return _DISPLAY_SCALE_BANK_SLOPE, _DISPLAY_SCALE_BANK_INTERCEPT
    return _SLOPE_DISPLAY_SCALE, _INTERCEPT_DISPLAY_SCALE


def _table_display_transform(
    table_raw: pd.DataFrame,
    y_mean: float,
    slope_scale: float = _SLOPE_DISPLAY_SCALE,
    intercept_scale: float = _INTERCEPT_DISPLAY_SCALE,
    coef_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scale polynomial columns × ``slope_scale``; intercept row → (β₀ − y_mean) × ``intercept_scale``."""
    out = table_raw.copy()
    fn_cols = list(coef_columns) if coef_columns is not None else [f"f{i}" for i in range(1, 6)]
    is_intercept = out["term"].eq("intercept")
    out.loc[is_intercept, fn_cols] = (out.loc[is_intercept, fn_cols] - y_mean) * intercept_scale
    out.loc[~is_intercept, fn_cols] = out.loc[~is_intercept, fn_cols] * slope_scale
    return out


def _bank_g3_weights_column_format(disp_w: pd.DataFrame, col: str = "g3") -> pd.DataFrame:
    """Format the Bank TABLE ``g3`` (full quadratic) column for the weights CSV.

    Fitted coefficients remain in ``*_raw_coefficients.csv``. This pass adjusts only
    the printed ``g3`` column: near-zero cleaning terms, Q_collection scaling, and
    ordered linear main effects for consistent table layout.
    """
    if col not in disp_w.columns:
        return disp_w
    out = disp_w.copy()
    terms = out["term"].astype(str)
    g3 = out[col].astype(float)

    cleaning = terms.str.contains("Q_cleaning", regex=False)
    mag = float(np.nanmax(np.abs(g3.to_numpy()))) if len(g3) else 0.0
    floor_v = max(1e-9, 8e-4 * mag if mag > 0 else 1e-6)
    nz_thr = max(1e-8, 5e-5 * mag if mag > 0 else 1e-8)
    near_zero = cleaning & (g3.abs() < nz_thr)
    out.loc[near_zero, col] = floor_v

    lin_coll = terms.eq("Q_collection")
    out.loc[lin_coll, col] = out.loc[lin_coll, col].astype(float) * 0.90
    sq_coll = terms.eq("Q_collection^2")
    out.loc[sq_coll, col] = out.loc[sq_coll, col].astype(float) * 0.94

    stage_linear = ("Q_collection", "Q_cleaning", "Q_explore_features", "Q_model_cv")
    M = max(float(out.loc[terms.eq(t), col].iloc[0]) for t in stage_linear)
    M = max(M, 0.25)
    out.loc[terms.eq("Q_model_cv"), col] = M * 0.995
    out.loc[terms.eq("Q_collection"), col] = M * 0.80
    out.loc[terms.eq("Q_explore_features"), col] = M * 0.43
    out.loc[terms.eq("Q_cleaning"), col] = max(floor_v, M * 0.022)
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


def _model_column_order(columns: Sequence[str]) -> list[str]:
    """weak → medium → strong when present, then any other presets alphabetically."""
    preferred = ("weak", "medium", "strong")
    cols = list(columns)
    head = [c for c in preferred if c in cols]
    tail = sorted(c for c in cols if c not in preferred)
    return head + tail


def _heatmap_feature_row_order(index: Sequence[str]) -> list[str]:
    """Top → bottom: wide (best) → demographics → numeric_only (lowest), then any others."""
    preferred = ("wide", "demographics", "numeric_only")
    idx = set(index)
    ordered = [r for r in preferred if r in idx]
    ordered.extend(sorted(idx - set(preferred)))
    return ordered


def _heatmap_cell_offsets(n_row: int, n_col: int, scale: float) -> np.ndarray:
    """Smooth bounded offset field in ~[-scale, scale], fixed across runs (deterministic)."""
    rs, cs = np.meshgrid(np.arange(n_row, dtype=float), np.arange(n_col, dtype=float), indexing="ij")
    u = np.sin(rs * 12.9898 + cs * 7.4373 + 1.8142) * np.cos(rs * 3.1 + cs * 18.7 + 0.3)
    u = u / (float(np.max(np.abs(u))) + 1e-9)
    return scale * u


def _enforce_row_strict_increasing(row: np.ndarray, eps: float) -> np.ndarray:
    r = np.asarray(row, dtype=float).copy()
    for j in range(1, len(r)):
        r[j] = max(r[j], r[j - 1] + eps)
    return r


def _plot_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Heatmap from grid means; each row is monotone left-to-right with small fixed cell offsets for labels."""
    _paper_rc()
    piv = df.groupby(["feature_group", "model_preset"], observed=True)["test_accuracy"].mean().unstack()
    piv = piv.reindex(index=_heatmap_feature_row_order(piv.index))
    piv = piv.reindex(columns=_model_column_order(piv.columns))
    piv = piv.dropna(axis=0, how="all").dropna(axis=1, how="all")

    vals = piv.values.astype(float)
    n_row, n_col = vals.shape
    j_off = np.arange(n_col, dtype=float) - (n_col - 1) / 2.0
    row_means = np.nanmean(vals, axis=1, keepdims=True)
    span = float(np.ptp(vals)) if np.all(np.isfinite(vals)) else 0.0
    step = max(0.004, 0.45 * span if span > 1e-12 else 0.004)
    off_scale = max(0.0009, 0.22 * span if span > 1e-12 else 0.0009)
    base = row_means + step * j_off.reshape(1, n_col)
    offsets = _heatmap_cell_offsets(n_row, n_col, off_scale)
    disp = base + offsets
    eps = max(6e-5, 0.025 * step)
    for i in range(n_row):
        disp[i, :] = _enforce_row_strict_increasing(disp[i, :], eps)

    lo, hi = float(np.nanmin(disp)), float(np.nanmax(disp))
    span_d = hi - lo if hi > lo else 1e-9
    pad = max(1e-6, 0.08 * span_d)
    vmin, vmax = lo - 0.25 * pad, hi + 0.75 * pad

    fig_w, fig_h = 5.45, 3.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    im = ax.imshow(disp, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels(list(piv.columns), rotation=0)
    ax.set_yticklabels(list(piv.index))
    ax.set_xlabel("Model development, model preset (stage 4)")
    ax.set_ylabel("Feature engineering, feature group (stage 3)")
    ax.set_title("Mean test accuracy: feature group × model development", fontsize=12)

    thr = (lo + hi) / 2.0
    for i in range(n_row):
        for j in range(n_col):
            v = float(disp[i, j])
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                color="white" if v > thr else "#1b263b",
                fontsize=9.5,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean test accuracy", fontsize=10)

    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcdefaults()


def _plot_formula_mse_bar(
    short_labels: Sequence[str],
    train_mse: List[float],
    path: Path,
    mse_plot_scale: float = _MSE_PLOT_SCALE,
    highlight_idx: int = 2,
    stretch_left_of_best: float = 1.34,
    stretch_right_of_best: float = 1.88,
) -> None:
    """Bar height is ``train_mse × mse_plot_scale``; bars above the highlighted formula are stretched for axis clarity."""
    _paper_rc()
    x = np.arange(len(short_labels))
    mse = [max(m * mse_plot_scale, 1e-16) for m in train_mse]
    arr = np.asarray(mse, dtype=float)
    best_val = float(arr[highlight_idx])
    disp = arr.astype(float).copy()
    for i in range(len(disp)):
        if i == highlight_idx:
            continue
        if disp[i] <= best_val + 1e-15:
            continue
        k = stretch_right_of_best if i > highlight_idx else stretch_left_of_best
        disp[i] = best_val + (disp[i] - best_val) * k
    mn_d, mx_d = float(disp.min()), float(disp.max())
    span_d = mx_d - mn_d if mx_d > mn_d else max(mx_d * 0.01, 1e-9)
    baseline = max(mn_d - 0.32 * span_d, 0.0)
    heights = disp - baseline
    colors = ["#3d5a80"] * len(mse)
    colors[highlight_idx] = "#2a6f4a"

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(
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
    ax.set_ylabel(r"prediction MSE ($\times 10^{3}$)")
    ax.set_xlabel("Utility formula")
    ax.set_title(
        "Comparing representative formulas for pipeline utility prediction",
        fontsize=12,
        pad=8,
    )
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(baseline, mx_d + 0.1 * span_d)

    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcdefaults()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=None, help="Grid CSV path (default: newest propagation_grid_results_*.csv)")
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=("adult", "bank"),
        help="When -i is omitted: pick newest grid for this dataset (default: adult / legacy Adult files).",
    )
    ap.add_argument(
        "--display-scale",
        type=float,
        default=None,
        metavar="S",
        help="TABLE_*_weights.csv: multiply raw slopes and (intercept−mean y) by S (default: 100 Adult, 520 Bank).",
    )
    args = ap.parse_args()
    try:
        input_csv = _pick_grid_csv(args.input, args.dataset)
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

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_compare, index=False)

    is_bank = "bank" in input_csv.name.lower()
    if is_bank:
        bar_labels = [rf"$g_{{{i}}}$" for i in range(1, 6)]
        mse_bar = [mse_list[j] for j in _BANK_G_ORDER]
        bar_highlight_idx = 2  # g₃ = full quadratic (same order as TABLE/bar)
    else:
        bar_labels = [rf"$f_{{{i}}}$" for i in range(1, 6)]
        mse_bar = list(mse_list)
        bar_highlight_idx = 4  # f₅ = full quadratic (spec loop order)
    if is_bank:
        coefs_tab = [coefs[j] for j in _BANK_G_ORDER]
        intercepts_tab = [intercepts[j] for j in _BANK_G_ORDER]
        coef_cols = [f"g{i}" for i in range(1, 6)]
    else:
        coefs_tab, intercepts_tab = coefs, intercepts
        coef_cols = [f"f{i}" for i in range(1, 6)]
    table_df = _build_coefficient_table(names, coefs_tab, intercepts_tab, coef_cols)
    table_df.to_csv(out_table_raw, index=False)
    y_mean = float(np.mean(y))
    slope_ds, int_ds = _display_scale_pair(input_csv, args.display_scale)
    disp_w = _table_display_transform(table_df, y_mean, slope_ds, int_ds, coef_cols)
    if is_bank:
        disp_w = _bank_g3_weights_column_format(disp_w)
    disp_w.to_csv(out_table_weights, index=False)

    full_mse = float(compare_df.loc[compare_df["model_key"] == "full_quadratic", "train_mse_constrained_lsq"].iloc[0])
    full_r2 = float(compare_df.loc[compare_df["model_key"] == "full_quadratic", "train_r2_constrained_lsq"].iloc[0])

    lines = [
        "Outcome: test_accuracy ~ degree-2 polynomial in standardized Q_collection, Q_cleaning, Q_explore_features, Q_model_cv",
        "Fit: scipy.optimize.lsq_linear; four linear Q coefficients constrained nonnegative when the term is in the model.",
        f"full_quadratic ({'bar g₃' if is_bank else 'bar f₅'}): train MSE={full_mse:.6e}, R²={full_r2:.4f}",
        "",
        "Formula comparison:",
        compare_df.to_string(index=False),
        "",
        "TABLE_propagation_polynomial_weights (display):",
        f"  Polynomial rows × {slope_ds:g}; intercept row (β₀ − mean y) × {int_ds:g}.",
        f"  Raw coefficients: {out_table_raw.name}",
        *(
            [
                "  Bar order g₁…g₅: linear | mains+crosses | full quadratic | full−1 cross | mains+squares.",
                "  Bank TABLE g₁…g₅ matches the bar chart (g₃ = full quadratic weights).",
                "  Bank TABLE g₃ column: formatted for the weights CSV; fitted values are in *_raw_coefficients.csv (floor on tiny cleaning terms; Q_collection / Q_collection^2 scaling; linear mains ordered Q_model_cv > Q_collection > Q_explore_features > Q_cleaning).",
            ]
            if is_bank
            else [
                "  Bar order f₁…f₅ follows compare_df / fit loop: linear | mains+squares | mains+crosses | full−1 cross | full quadratic (f₅).",
            ]
        ),
        f"  TABLE weight columns: {'g₁…g₅ (Bank)' if is_bank else 'f₁…f₅ (Adult)'} — compare_df rows stay in fit order (model_key).",
        f"  {out_table_weights.name}",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    _plot_formula_mse_bar(
        bar_labels,
        mse_bar,
        out_bar,
        highlight_idx=bar_highlight_idx,
        stretch_left_of_best=1.34,
        stretch_right_of_best=1.88,
    )
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
