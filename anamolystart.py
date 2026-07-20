#!/usr/bin/env python3
"""Quality-control pipeline for wavelength-dependent PMT QE spectra.

The pipeline combines robust population statistics, engineered shape features,
PCA reconstruction error, and Isolation Forest scores. It writes QC tables,
leaderboards, and diagnostic plots.
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# CONFIG


@dataclass
class PipelineConfig:
    # wavelength grid
    wl_min: float = 280.0
    wl_max: float = 700.0
    wl_step: float = 5.0
    eps: float = 1e-6

    # soft physical sanity range
    qe_min_soft: float = -3.0
    qe_max_soft: float = 55.0
    range_p_lo: float = 1.0
    range_p_hi: float = 99.0

    # transition region
    boundary_center: float = 400.0
    boundary_halfwidth: float = 10.0

    # curvature / kink
    curvature_z_thresh: float = 7.0

    # PCA
    pca_var_keep: float = 0.98
    pca_inlier_keep: float = 0.85

    # Isolation Forest
    if_n_estimators: int = 500
    if_contamination: str = "auto"
    if_random_state: int = 42

    # degradation band
    degr_band_a: float = 300.0
    degr_band_b: float = 500.0

    # reporting
    topk_worst: int = 15
    topk_best: int = 15
    topk_degraded: int = 15
    topk_if: int = 15

    outdir: str = "qe_hybrid_ml_qc_outputs"
    plots_union_dirname: str = "plots_review_priority"

    # Shape-score weights
    w_pca: float = 0.35
    w_iforest: float = 0.25
    w_boundary: float = 0.15
    w_kink: float = 0.10
    w_roughness: float = 0.10
    w_range: float = 0.05

    # degradation weights
    w_low_qe_peak: float = 0.10
    w_low_qe_band: float = 0.20

    # final QC priority score
    degradation_in_review_weight: float = 0.40


# COLUMN OPTIONS

WAVELENGTH_COLUMNS = [
    "Wavelength (nm)",
    "wavelength (nm)",
    "Actual WL (nm)",
    "Requested WL (nm)",
]

QE_COLUMNS = [
    "QE_PMT_corrected (%)",
    "QE_PMT (%)",
    "QE PMT (%)",
    "QE_PMT_corrected",
    "QE_PMT_uncorrected (%)",
]


# BASIC HELPERS


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    )
    return df


def pick_column(df: pd.DataFrame, candidates: List[str]):
    col_map = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().strip()
        if key in col_map:
            return col_map[key]

    return None


def detect_wavelength_and_qe_columns(df: pd.DataFrame):
    """Detect wavelength and QE columns, preferring corrected QE values."""
    df = clean_cols(df)

    wl_col = pick_column(df, WAVELENGTH_COLUMNS)
    qe_col = pick_column(df, QE_COLUMNS)

    if wl_col is None:
        for c in df.columns:
            cl = c.lower()
            if ("wavelength" in cl) or ("actual wl" in cl) or ("requested wl" in cl):
                wl_col = c
                break

    if qe_col is None:
        for c in df.columns:
            cl = c.lower()
            if ("qe" in cl) and ("%" in c):
                qe_col = c
                break

    if qe_col is None:
        for c in df.columns:
            if "qe" in c.lower():
                qe_col = c
                break

    if wl_col is None or qe_col is None:
        raise ValueError(
            "Could not detect wavelength/QE columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    return wl_col, qe_col


def mad(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.nanmedian(x, axis=axis)
    return np.nanmedian(np.abs(x - med), axis=axis)


def rank_pct(s: pd.Series, ascending=True) -> pd.Series:
    """Return percentile ranks after replacing missing values with the median."""
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().sum() == 0:
        return pd.Series(np.zeros(len(s2)), index=s2.index)
    return s2.fillna(s2.median()).rank(pct=True, ascending=ascending)


def safe_ratio(a, b, eps=1e-9):
    return a / (b + eps)


# ID PARSING / LOADING


def parse_pmt_id_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem_u = stem.upper()

    if ("REFERENCE" in stem_u) or re.search(r"\bREF\b", stem_u):
        return "TM0007"

    m = re.search(r"ST\D*0*(\d+)", stem_u)
    if m:
        return f"ST{int(m.group(1)):04d}"

    m = re.search(r"TM\D*0*(\d+)", stem_u)
    if m:
        return f"TM{int(m.group(1)):04d}"

    safe = re.sub(r"\s+", "_", stem.strip())
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "", safe)

    return safe


def extract_timestamp_from_filename(filename: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", filename)
    if m:
        return m.group(1)
    return ""


def load_qe_curves_from_dir(
    qe_dir: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Load QE curves and retain the latest timestamp for duplicate PMT IDs."""
    curves: Dict[str, pd.DataFrame] = {}
    meta: Dict[str, Dict] = {}

    files = sorted(glob.glob(os.path.join(qe_dir, "*.csv")))

    if len(files) == 0:
        raise SystemExit(f"No CSV files found in: {qe_dir}")

    records = []
    skipped = []

    for fp in files:
        base = os.path.basename(fp)
        pid = parse_pmt_id_from_filename(base)
        timestamp = extract_timestamp_from_filename(base)

        try:
            df = pd.read_csv(fp)
            df = clean_cols(df)

            wl_col, qe_col = detect_wavelength_and_qe_columns(df)

            df2 = pd.DataFrame(
                {
                    "wl_nm": pd.to_numeric(df[wl_col], errors="coerce"),
                    "qe": pd.to_numeric(df[qe_col], errors="coerce"),
                }
            )

            df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df2) < 5:
                skipped.append(
                    {
                        "source_file": base,
                        "pmt_id": pid,
                        "reason": "Too few valid points",
                    }
                )
                continue

            records.append(
                {
                    "pmt_id": pid,
                    "timestamp": timestamp,
                    "source_file": base,
                    "source_path": fp,
                    "wl_col_used": wl_col,
                    "qe_col_used": qe_col,
                    "data": df2[["wl_nm", "qe"]].copy(),
                }
            )

        except Exception as e:
            skipped.append(
                {
                    "source_file": base,
                    "pmt_id": pid,
                    "reason": str(e),
                }
            )

    if not records:
        raise SystemExit(
            "No valid QE curves loaded. Check column names and folder path."
        )

    records_df = pd.DataFrame(records)
    records_df = records_df.sort_values(
        by=["pmt_id", "timestamp"],
        ascending=[True, True],
    )

    latest_df = records_df.drop_duplicates(
        subset=["pmt_id"],
        keep="last",
    )

    for _, row in latest_df.iterrows():
        pid = row["pmt_id"]

        curves[pid] = row["data"]

        meta[pid] = {
            "pmt_id": pid,
            "source_file": row["source_file"],
            "source_path": row["source_path"],
            "wl_col_used": row["wl_col_used"],
            "qe_col_used": row["qe_col_used"],
        }

    print(f"[INFO] Valid unique PMT curves loaded: {len(curves)}")

    if skipped:
        os.makedirs("qe_qc_loader_logs", exist_ok=True)
        pd.DataFrame(skipped).to_csv(
            "qe_qc_loader_logs/skipped_qe_files.csv",
            index=False,
        )
        print(
            f"[WARN] Skipped {len(skipped)} files. Log saved to qe_qc_loader_logs/skipped_qe_files.csv"
        )

    return curves, meta


# INTERPOLATION / CONSENSUS


def interpolate_curve(wl: np.ndarray, y: np.ndarray, wl_grid: np.ndarray) -> np.ndarray:
    wl = wl.astype(float)
    y = y.astype(float)

    mask = np.isfinite(wl) & np.isfinite(y)
    wl = wl[mask]
    y = y[mask]

    if wl.size < 2:
        return np.full_like(wl_grid, np.nan, dtype=float)

    idx = np.argsort(wl)
    wl, y = wl[idx], y[idx]

    _, uniq_idx = np.unique(wl, return_index=True)
    wl, y = wl[uniq_idx], y[uniq_idx]

    f = interp1d(
        wl,
        y,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    return f(wl_grid)


def consensus_stats(
    Q: np.ndarray, cfg: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Q, axis=0)
    spread = mad(Q, axis=0) + cfg.eps
    return med, spread


def band_median(arr: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    return float(np.nanmedian(arr[mask]))


def mean_band(wl_grid: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    m = (wl_grid >= a) & (wl_grid <= b) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.nanmean(y[m]))


def auc_band(wl_grid: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    m = (wl_grid >= a) & (wl_grid <= b) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.trapezoid(y[m], wl_grid[m]))


def slope_band(wl_grid: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    m = (wl_grid >= a) & (wl_grid <= b) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan

    x = wl_grid[m]
    yy = y[m]

    p = np.polyfit(x, yy, deg=1)
    return float(p[0])


def fwhm(wl_grid: np.ndarray, y: np.ndarray) -> float:
    if np.isfinite(y).sum() < 10:
        return np.nan

    ymax = np.nanmax(y)

    if not np.isfinite(ymax) or ymax <= 0:
        return np.nan

    half = 0.5 * ymax
    above = y >= half

    if above.sum() < 2:
        return np.nan

    wl_span = wl_grid[above]
    return float(wl_span[-1] - wl_span[0])


# RESIDUAL / LOW-HIGH SCORES


def residual_scores(wl_grid, qe_grid, med, spread, cfg) -> Dict[str, float]:
    r = qe_grid - med
    z = np.abs(r) / spread
    bw = cfg.boundary_halfwidth

    return {
        "S_global": float(np.nanmedian(z)),
        "S_max": float(np.nanmax(z)),
        "S_boundary": band_median(
            z,
            (wl_grid >= cfg.boundary_center - bw)
            & (wl_grid <= cfg.boundary_center + bw),
        ),
        "S_280_400": band_median(z, (wl_grid >= 280) & (wl_grid <= 400)),
        "S_400_700": band_median(z, (wl_grid >= 400) & (wl_grid <= 700)),
    }


def low_qe_scores(wl_grid, qe_grid, med, spread) -> Dict[str, float]:
    r_low = np.maximum(0.0, med - qe_grid)
    z_low = r_low / spread

    return {
        "S_low_global": float(np.nanmedian(z_low)),
        "S_low_280_400": band_median(z_low, (wl_grid >= 280) & (wl_grid <= 400)),
        "S_low_400_700": band_median(z_low, (wl_grid >= 400) & (wl_grid <= 700)),
    }


def high_qe_scores(wl_grid, qe_grid, med, spread) -> Dict[str, float]:
    r_hi = np.maximum(0.0, qe_grid - med)
    z_hi = r_hi / spread

    return {
        "S_high_global": float(np.nanmedian(z_hi)),
    }


# SHAPE FEATURES


def curvature_raw(qe_grid: np.ndarray, wl_step: float) -> float:
    y = qe_grid.copy()

    if np.isfinite(y).sum() < 5:
        return np.nan

    dy = np.gradient(y, wl_step)
    d2y = np.gradient(dy, wl_step)

    return float(np.nanmax(np.abs(d2y)))


def total_variation(y: np.ndarray) -> float:
    return float(np.nansum(np.abs(np.diff(y))))


def boundary_step_metric(
    wl_grid: np.ndarray,
    y: np.ndarray,
    center=400.0,
    w=10.0,
) -> float:
    left = (wl_grid >= center - w) & (wl_grid < center) & np.isfinite(y)
    right = (wl_grid > center) & (wl_grid <= center + w) & np.isfinite(y)

    if left.sum() < 2 or right.sum() < 2:
        return np.nan

    pl = np.polyfit(wl_grid[left], y[left], 1)
    pr = np.polyfit(wl_grid[right], y[right], 1)

    return float(np.polyval(pr, center) - np.polyval(pl, center))


def peak_qe_and_wl(wl_grid: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if np.isfinite(y).sum() < 3:
        return np.nan, np.nan

    i = int(np.nanargmax(y))

    return float(y[i]), float(wl_grid[i])


def tail_rise_metric(wl_grid: np.ndarray, y: np.ndarray, start=580.0) -> float:
    m = (wl_grid >= start) & np.isfinite(y)

    if m.sum() < 2:
        return np.nan

    yy = y[m]
    return float(yy[-1] - yy[0])


def extract_extra_shape_features(
    wl_grid: np.ndarray,
    y: np.ndarray,
    cfg: PipelineConfig,
) -> Dict[str, float]:

    peak_qe, peak_wl = peak_qe_and_wl(wl_grid, y)

    auc_uv = auc_band(wl_grid, y, 280, 400)
    auc_vis = auc_band(wl_grid, y, 400, 700)

    return {
        "peak_qe": peak_qe,
        "peak_wl": peak_wl,
        "fwhm": fwhm(wl_grid, y),
        "auc_280_400": auc_uv,
        "auc_400_700": auc_vis,
        "auc_uv_vis_ratio": safe_ratio(auc_uv, auc_vis),
        "slope_450_550": slope_band(wl_grid, y, 450, 550),
        "slope_550_650": slope_band(wl_grid, y, 550, 650),
        "tail_rise_580_700": tail_rise_metric(wl_grid, y, start=580.0),
        "qe_mean_band": mean_band(wl_grid, y, cfg.degr_band_a, cfg.degr_band_b),
    }


# PCA


def pca_recon_errors_two_pass(Q: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Two-pass PCA:
    1. Fill missing values by wavelength median.
    2. Robust scale.
    3. Select central inlier fraction.
    4. Fit PCA on inliers.
    5. Compute reconstruction error for all PMTs.
    """
    Qf = Q.copy()

    col_med = np.nanmedian(Qf, axis=0)
    global_med = np.nanmedian(Qf)
    col_med = np.where(np.isfinite(col_med), col_med, global_med)

    inds = np.where(~np.isfinite(Qf))
    Qf[inds] = np.take(col_med, inds[1])

    scaler = RobustScaler()
    Z = scaler.fit_transform(Qf)

    z_med = np.median(Z, axis=0)
    pre = np.linalg.norm(Z - z_med[None, :], axis=1)

    keep_n = int(np.ceil(cfg.pca_inlier_keep * len(pre)))
    keep_n = max(2, keep_n)

    inlier_idx = np.argsort(pre)[:keep_n]
    Z_in = Z[inlier_idx]

    pca = PCA(n_components=cfg.pca_var_keep, svd_solver="full")
    pca.fit(Z_in)

    X_all = pca.transform(Z)
    Zhat = pca.inverse_transform(X_all)

    err = np.linalg.norm(Z - Zhat, axis=1)

    return err


# ISOLATION FOREST


def isolation_forest_scores(
    feature_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Isolation Forest on engineered SHAPE/QC features.

    Important:
    This is not used as the only decision criterion.
    It is an additional unsupervised anomaly indicator.
    """
    X = feature_df.copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)

    scaler = RobustScaler()
    Z = scaler.fit_transform(X.values)

    iso = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        contamination=cfg.if_contamination,
        random_state=cfg.if_random_state,
    )

    iso.fit(Z)

    score = -iso.decision_function(Z)

    return score


# PLOTTING


def save_flag_plot(
    pmt_id: str,
    wl_grid: np.ndarray,
    qe: np.ndarray,
    med: np.ndarray,
    spread: np.ndarray,
    outpath: str,
    title_prefix: str = "QE QC",
    subtitle: str = "",
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        wl_grid,
        qe,
        linewidth=1.6,
        label="PMT",
    )

    plt.plot(
        wl_grid,
        med,
        linewidth=2.0,
        label="Population median",
    )

    plt.fill_between(
        wl_grid,
        med - 2 * spread,
        med + 2 * spread,
        alpha=0.2,
        label="Median ± 2 MAD",
    )

    plt.axvline(
        400, linestyle="--", linewidth=1.0, alpha=0.7, label="400 nm transition"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("QE (%)")

    title = f"{title_prefix}: {pmt_id}"
    if subtitle:
        title += f"\n{subtitle}"

    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# MAIN PIPELINE


def run_pipeline(
    curves: Dict[str, pd.DataFrame],
    meta: Dict[str, Dict],
    cfg: PipelineConfig,
) -> pd.DataFrame:

    os.makedirs(cfg.outdir, exist_ok=True)

    print("[INFO] Saving outputs to:", os.path.abspath(cfg.outdir))

    wl_grid = np.arange(
        cfg.wl_min,
        cfg.wl_max + 0.5 * cfg.wl_step,
        cfg.wl_step,
    )

    pmt_ids = list(curves.keys())
    pid_to_i = {pid: i for i, pid in enumerate(pmt_ids)}

    Q_list: List[np.ndarray] = []

    for pid in pmt_ids:
        df = curves[pid]

        qe_i = interpolate_curve(
            df["wl_nm"].to_numpy(),
            df["qe"].to_numpy(),
            wl_grid,
        )

        Q_list.append(qe_i)

    Q = np.vstack(Q_list)

    med, spread = consensus_stats(Q, cfg)

    rows = []

    for i, _pid in enumerate(pmt_ids):
        qe = Q[i]
        q = qe[np.isfinite(qe)]

        if q.size > 0:
            lo = np.nanpercentile(q, cfg.range_p_lo)
            hi = np.nanpercentile(q, cfg.range_p_hi)

            range_oob = bool(lo < cfg.qe_min_soft or hi > cfg.qe_max_soft)
        else:
            range_oob = True

        row = {}

        row.update(residual_scores(wl_grid, qe, med, spread, cfg))
        row.update(low_qe_scores(wl_grid, qe, med, spread))
        row.update(high_qe_scores(wl_grid, qe, med, spread))

        row["RANGE_OOB"] = range_oob

        row["curvature_raw"] = curvature_raw(qe, cfg.wl_step)
        row["total_variation"] = total_variation(qe)
        row["boundary_step"] = boundary_step_metric(
            wl_grid,
            qe,
            cfg.boundary_center,
            cfg.boundary_halfwidth,
        )

        row.update(extract_extra_shape_features(wl_grid, qe, cfg))

        rows.append(row)

    out = pd.DataFrame(rows, index=pmt_ids)

    # Robust curvature z-score / kink flag

    curv = out["curvature_raw"].to_numpy(dtype=float)

    curv_med = np.nanmedian(curv)
    curv_mad = np.nanmedian(np.abs(curv - curv_med)) + cfg.eps

    out["curvature_z_robust"] = (curv - curv_med) / curv_mad
    out["curvature_z_abs"] = np.abs(out["curvature_z_robust"])

    out["kink_flag"] = out["curvature_z_abs"] > cfg.curvature_z_thresh

    # PCA

    out["pca_recon_err"] = pca_recon_errors_two_pass(Q, cfg)

    # Isolation Forest on shape/QC features

    out["boundary_step_abs"] = np.abs(out["boundary_step"])

    if_feature_cols = [
        "pca_recon_err",
        "S_global",
        "S_boundary",
        "boundary_step_abs",
        "curvature_z_abs",
        "total_variation",
        "peak_wl",
        "fwhm",
        "auc_uv_vis_ratio",
        "slope_450_550",
        "slope_550_650",
        "tail_rise_580_700",
    ]

    if_feature_df = out[if_feature_cols].copy()

    out["isoforest_score"] = isolation_forest_scores(if_feature_df, cfg)
    out["isoforest_rank"] = rank_pct(out["isoforest_score"], ascending=True)

    # Metadata

    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    out = meta_df.join(out, how="right")

    # Shape score

    rpca = rank_pct(out["pca_recon_err"], ascending=True)
    rif = rank_pct(out["isoforest_score"], ascending=True)
    rbnd = rank_pct(out["S_boundary"], ascending=True)
    rkink = out["kink_flag"].astype(int)
    rrough = rank_pct(out["total_variation"], ascending=True)
    rrange = out["RANGE_OOB"].astype(int)

    out["contrib_pca"] = cfg.w_pca * rpca
    out["contrib_iforest"] = cfg.w_iforest * rif
    out["contrib_boundary"] = cfg.w_boundary * rbnd
    out["contrib_kink"] = cfg.w_kink * rkink
    out["contrib_roughness"] = cfg.w_roughness * rrough
    out["contrib_range"] = cfg.w_range * rrange

    out["shape_score"] = (
        out["contrib_pca"]
        + out["contrib_iforest"]
        + out["contrib_boundary"]
        + out["contrib_kink"]
        + out["contrib_roughness"]
        + out["contrib_range"]
    )

    # Degradation score

    r_peak_low = 1.0 - rank_pct(out["peak_qe"], ascending=True)
    r_band_low = 1.0 - rank_pct(out["qe_mean_band"], ascending=True)

    out["contrib_low_peak"] = cfg.w_low_qe_peak * r_peak_low
    out["contrib_low_band"] = cfg.w_low_qe_band * r_band_low

    out["degradation_score"] = out["contrib_low_peak"] + out["contrib_low_band"]

    # Final QC priority score

    out["qc_priority_score"] = (
        out["shape_score"] + cfg.degradation_in_review_weight * out["degradation_score"]
    )

    out["best_score_combined"] = 1.0 - rank_pct(
        out["qc_priority_score"], ascending=True
    )

    # Quality tiers

    p_shape = rank_pct(out["shape_score"], ascending=True)

    out["quality_tier"] = np.select(
        [
            (out["kink_flag"] == 0) & (out["RANGE_OOB"] == 0) & (p_shape < 0.25),
            (out["kink_flag"] == 0) & (out["RANGE_OOB"] == 0) & (p_shape < 0.45),
            (p_shape < 0.70),
        ],
        [
            "GOLD",
            "SILVER",
            "BRONZE",
        ],
        default="REVIEW",
    )

    out["measurement_trust"] = np.where(
        out["quality_tier"].isin(["GOLD", "SILVER", "BRONZE"]),
        "CLEAN",
        "MEAS_SUSPECT",
    )

    # Primary driver

    driver_cols = {
        "PCA_SHAPE": "contrib_pca",
        "ISOFOREST_SHAPE": "contrib_iforest",
        "BOUNDARY": "contrib_boundary",
        "KINK": "contrib_kink",
        "ROUGHNESS": "contrib_roughness",
        "RANGE": "contrib_range",
        "LOW_QE_PEAK": "contrib_low_peak",
        "LOW_QE_BAND": "contrib_low_band",
    }

    def primary_driver(row):
        vals = {
            name: row[col]
            for name, col in driver_cols.items()
            if col in row.index and np.isfinite(row[col])
        }
        return max(vals, key=vals.get)

    out["primary_driver"] = out.apply(primary_driver, axis=1)

    out["primary_family"] = np.where(
        out["shape_score"]
        >= cfg.degradation_in_review_weight * out["degradation_score"],
        "SHAPE",
        "DEGRADATION",
    )

    # Save outputs

    out_csv = os.path.join(cfg.outdir, "anomaly_table_full.csv")
    out.to_csv(out_csv, index=True)

    ref = pd.DataFrame(
        {
            "wl_nm": wl_grid,
            "median_qe": med,
            "mad_qe": spread,
        }
    )

    ref.to_csv(
        os.path.join(cfg.outdir, "reference_median_mad.csv"),
        index=False,
    )

    shape_bad = out.sort_values("shape_score", ascending=False)
    if_bad = out.sort_values("isoforest_score", ascending=False)
    degr_bad = out.sort_values("degradation_score", ascending=False)
    review_priority = out.sort_values("qc_priority_score", ascending=False)
    best = out.sort_values("best_score_combined", ascending=False)

    shape_bad.to_csv(
        os.path.join(cfg.outdir, "leaderboard_shape_anomaly.csv"),
        index=True,
    )

    if_bad.to_csv(
        os.path.join(cfg.outdir, "leaderboard_isolation_forest.csv"),
        index=True,
    )

    degr_bad.to_csv(
        os.path.join(cfg.outdir, "leaderboard_degradation.csv"),
        index=True,
    )

    review_priority.to_csv(
        os.path.join(cfg.outdir, "leaderboard_review_priority.csv"),
        index=True,
    )

    best.to_csv(
        os.path.join(cfg.outdir, "leaderboard_best_combined.csv"),
        index=True,
    )

    # Save plots

    plot_dir_best = os.path.join(cfg.outdir, "plots_best")
    os.makedirs(plot_dir_best, exist_ok=True)

    for pid in best.head(cfg.topk_best).index:
        i = pid_to_i[pid]

        subtitle = (
            f"tier={out.loc[pid, 'quality_tier']}, "
            f"shape={out.loc[pid, 'shape_score']:.3f}, "
            f"degr={out.loc[pid, 'degradation_score']:.3f}"
        )

        save_flag_plot(
            pid,
            wl_grid,
            Q[i],
            med,
            spread,
            os.path.join(plot_dir_best, f"best_{pid}.png"),
            title_prefix="Best spectrum",
            subtitle=subtitle,
        )

    plot_dir_union = os.path.join(cfg.outdir, cfg.plots_union_dirname)
    os.makedirs(plot_dir_union, exist_ok=True)

    candidate_lists = [
        list(review_priority.head(cfg.topk_worst).index),
        list(degr_bad.head(cfg.topk_degraded).index),
        list(if_bad.head(cfg.topk_if).index),
    ]

    plot_ids = []
    seen = set()

    for ids in candidate_lists:
        for pid in ids:
            if pid not in seen:
                plot_ids.append(pid)
                seen.add(pid)

    for pid in plot_ids:
        i = pid_to_i[pid]

        subtitle = (
            f"driver={out.loc[pid, 'primary_driver']}, "
            f"tier={out.loc[pid, 'quality_tier']}, "
            f"shape={out.loc[pid, 'shape_score']:.3f}, "
            f"IF={out.loc[pid, 'isoforest_score']:.3f}, "
            f"degr={out.loc[pid, 'degradation_score']:.3f}"
        )

        save_flag_plot(
            pid,
            wl_grid,
            Q[i],
            med,
            spread,
            os.path.join(plot_dir_union, f"qc_{pid}.png"),
            title_prefix="QC review candidate",
            subtitle=subtitle,
        )

    print(f"[OK] Saved full table: {out_csv}")
    print("[OK] Saved leaderboards:")
    print("     - leaderboard_shape_anomaly.csv")
    print("     - leaderboard_isolation_forest.csv")
    print("     - leaderboard_degradation.csv")
    print("     - leaderboard_review_priority.csv")
    print("     - leaderboard_best_combined.csv")
    print(f"[OK] Best plots -> {plot_dir_best}/")
    print(f"[OK] Review-priority plots ({len(plot_ids)} total) -> {plot_dir_union}/")

    return out


# ENTRYPOINT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qe_dir", help="Directory containing QE CSV files.")
    parser.add_argument(
        "--outdir",
        default="qe_hybrid_ml_qc_outputs_corrected",
        help="Directory for QC tables and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    QE_DIR = args.qe_dir

    cfg = PipelineConfig(
        outdir=args.outdir,
        topk_worst=15,
        topk_best=15,
        topk_degraded=15,
        topk_if=15,
    )

    curves, meta = load_qe_curves_from_dir(QE_DIR)

    tm_ids = [k for k in curves.keys() if k.upper().startswith("TM")]

    print(f"[INFO] Loaded {len(curves)} curves from: {QE_DIR}")
    print("[INFO] Parsed TM IDs:", tm_ids)

    out = run_pipeline(curves, meta, cfg)

    ref_id = "TM0007"

    if ref_id in out.index:
        print("\n=== REFERENCE PMT SCORE: TM0007 ===")

        cols = [
            "shape_score",
            "isoforest_score",
            "degradation_score",
            "qc_priority_score",
            "quality_tier",
            "measurement_trust",
            "primary_driver",
            "pca_recon_err",
            "S_boundary",
            "curvature_z_robust",
            "total_variation",
            "peak_qe",
            "peak_wl",
            "qe_mean_band",
            "qe_col_used",
            "source_file",
        ]

        print(out.loc[ref_id, [c for c in cols if c in out.columns]])

    else:
        print(
            "\n[WARN] TM0007 not found. "
            "Check reference filename contains 'REFERENCE', 'REF', or TM digits."
        )

    print("\n=== TOP 15 REVIEW-PRIORITY PMTs ===")

    print(
        out.sort_values("qc_priority_score", ascending=False).head(15)[
            [
                "qc_priority_score",
                "primary_family",
                "primary_driver",
                "shape_score",
                "isoforest_score",
                "degradation_score",
                "quality_tier",
                "measurement_trust",
                "peak_qe",
                "qe_mean_band",
                "qe_col_used",
            ]
        ]
    )

    print("\n=== TOP 15 ISOLATION-FOREST SHAPE ANOMALIES ===")

    print(
        out.sort_values("isoforest_score", ascending=False).head(15)[
            [
                "isoforest_score",
                "shape_score",
                "quality_tier",
                "primary_driver",
                "pca_recon_err",
                "S_boundary",
                "boundary_step",
                "curvature_z_robust",
                "total_variation",
                "peak_qe",
                "peak_wl",
            ]
        ]
    )

    print("\n=== TOP 15 MOST DEGRADED LOW-QE PMTs ===")

    print(
        out.sort_values("degradation_score", ascending=False).head(15)[
            [
                "degradation_score",
                "peak_qe",
                "qe_mean_band",
                "shape_score",
                "isoforest_score",
                "quality_tier",
                "measurement_trust",
                "qe_col_used",
            ]
        ]
    )
