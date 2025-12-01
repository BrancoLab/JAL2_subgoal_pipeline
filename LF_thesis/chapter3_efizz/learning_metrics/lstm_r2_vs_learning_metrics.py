"""Correlate LSTM A/B test R^2 with session-level learning metrics (scatter by metric)."""

from __future__ import annotations

from pathlib import Path
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ------------------- PATHS / SETTINGS -------------------
RESULTS_PKL = Path(r"Z:\Laurence\thesis\efizz_chapter\LSTM_results.pkl")
LEARNING_METRICS_PATH = Path(r"Z:\Laurence\thesis\efizz_chapter\outputs\learning_metrics_per_condition.csv")
OUT_DIR = Path(r"Z:\Laurence\thesis\efizz_chapter\plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["barrier_pre_flip", "barrier_post_flip"]
TARGET_MAP = {"h_preflipbar_a": "A", "h_postflipbar_a": "B"}
OUTLIER_Z = 1.96
IQR_FACTOR = 1.5
MICE_GROUPS = {
    "JAL6": ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may"],
    "JAL4": ["JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"],
    "JAL5": ["JAL005_8thSept", "JAL005_21stSept"],
}
DEFAULT_METRICS = [
    # "homings_count",
    # "escapes_count",
    "homings_mean_speed",
    "escapes_mean_speed",
    "homings_mean_efficiency",
    "escapes_mean_efficiency",
]


def load_lstm_df() -> pd.DataFrame:
    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)
    df = pd.DataFrame(results, columns=["session", "condition", "target", "test_r2"])
    df = df[df["condition"].isin(CONDITIONS)]
    df = df[df["target"].isin(TARGET_MAP)]
    df["target_label"] = df["target"].map(TARGET_MAP)
    return df


def load_learning_metrics() -> pd.DataFrame:
    df = pd.read_csv(LEARNING_METRICS_PATH)
    df = df[df["condition"].isin(CONDITIONS)].copy()
    return df


def select_metrics(df: pd.DataFrame) -> list[str]:
    available = [m for m in DEFAULT_METRICS if m in df.columns]
    if available:
        return available
    numeric_cols = [
        col
        for col, dtype in df.dtypes.items()
        if col not in {"session", "condition"} and np.issubdtype(dtype, np.number)
    ]
    return numeric_cols


def session_to_mouse(session: str) -> str:
    for mouse, sessions in MICE_GROUPS.items():
        if session in sessions:
            return mouse
    return "Unknown"


def prepare_merged_df() -> tuple[pd.DataFrame, list[str]]:
    lstm_df = load_lstm_df()
    metrics_df = load_learning_metrics()
    metrics = select_metrics(metrics_df)
    if not metrics:
        raise RuntimeError("No numeric learning metrics found to plot.")

    merged = metrics_df.merge(lstm_df, on=["session", "condition"], how="inner")
    merged = merged.dropna(subset=metrics + ["test_r2", "target_label"])
    merged["mouse"] = merged["session"].apply(session_to_mouse)
    return merged, metrics


def build_ab_diff_df(lstm_df: pd.DataFrame, metrics_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Create dataframe with A-B test R^2 difference per session/condition merged with metrics."""
    rows = []
    for cond in CONDITIONS:
        df_cond = lstm_df[lstm_df["condition"] == cond]
        pivot = df_cond.pivot_table(index="session", columns="target_label", values="test_r2")
        if {"A", "B"}.issubset(pivot.columns):
            pivot = pivot.dropna(subset=["A", "B"])
            diff = pivot["A"] - pivot["B"]
            for sess, val in diff.items():
                rows.append({"session": sess, "condition": cond, "ab_diff": val})
    if not rows:
        return pd.DataFrame()
    diff_df = pd.DataFrame(rows)
    merged = metrics_df.merge(diff_df, on=["session", "condition"], how="inner")
    merged = merged.dropna(subset=metrics + ["ab_diff"])
    return merged


def plot_scatter_by_metric(df: pd.DataFrame, metrics: list[str]):
    colors = {
        ("barrier_pre_flip", "A"): "#0984e3",
        ("barrier_pre_flip", "B"): "#6c5ce7",
        ("barrier_post_flip", "A"): "#d63031",
        ("barrier_post_flip", "B"): "#e17055",
    }
    markers = {"barrier_pre_flip": "o", "barrier_post_flip": "s"}
    cond_labels = {
        "barrier_pre_flip": "Pre flip",
        "barrier_post_flip": "Post flip",
    }

    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        stat_lines = []
        for cond in CONDITIONS:
            for target in sorted(df["target_label"].unique()):
                mask = (df["condition"] == cond) & (df["target_label"] == target)
                if not mask.any():
                    continue
                x = df.loc[mask, metric].to_numpy(dtype=float)
                y = df.loc[mask, "test_r2"].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                x = x[finite]
                y = y[finite]
                if x.size == 0:
                    continue
                ax.scatter(
                    x,
                    y,
                    label=f"{cond_labels.get(cond, cond)} {target}",
                    marker=markers.get(cond, "o"),
                    color=colors.get((cond, target), "#636e72"),
                    alpha=0.75,
                )
                if x.size >= 2 and np.nanmax(x) > np.nanmin(x):
                    coeffs = np.polyfit(x, y, deg=1)
                    line_x = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                    line_y = np.polyval(coeffs, line_x)
                    ax.plot(line_x, line_y, color=colors.get((cond, target), "#636e72"), linestyle="-", linewidth=1.2, alpha=0.9)

            cond_mask = df["condition"] == cond
            x_all = df.loc[cond_mask, metric].to_numpy(dtype=float)
            y_all = df.loc[cond_mask, "test_r2"].to_numpy(dtype=float)
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            x_all = x_all[finite]
            y_all = y_all[finite]
            if x_all.size >= 3 and np.nanmax(x_all) > np.nanmin(x_all):
                try:
                    r_val, p_val = pearsonr(x_all, y_all)
                    stat_lines.append(f"{cond_labels.get(cond, cond)}: r={r_val:.2f}, p={p_val:.3f}")
                except Exception:
                    pass

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("LSTM test R^2")
        ax.set_title(f"{metric} vs R^2")
        if stat_lines:
            ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)

    for j in range(n_metrics, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Learning metrics vs LSTM A/B R^2 (pre/post flip)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    out_path = OUT_DIR / "LSTM_R2_vs_learning_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved scatter grid to {out_path}")


def plot_scatter_by_metric_pooled(df: pd.DataFrame, metrics: list[str]):
    """Scatter per metric pooling A/B together with one line per condition."""
    cond_colors = {"barrier_pre_flip": "#0984e3", "barrier_post_flip": "#d63031"}
    target_markers = {"A": "o", "B": "s"}
    cond_labels = {"barrier_pre_flip": "Pre flip", "barrier_post_flip": "Post flip"}

    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        stat_lines = []
        for cond in CONDITIONS:
            cond_mask = df["condition"] == cond
            if not cond_mask.any():
                continue
            x_all = df.loc[cond_mask, metric].to_numpy(dtype=float)
            y_all = df.loc[cond_mask, "test_r2"].to_numpy(dtype=float)
            t_all = df.loc[cond_mask, "target_label"].to_numpy()
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            x_all = x_all[finite]
            y_all = y_all[finite]
            t_all = t_all[finite]
            if x_all.size == 0:
                continue
            for tgt in np.unique(t_all):
                tgt_mask = t_all == tgt
                ax.scatter(
                    x_all[tgt_mask],
                    y_all[tgt_mask],
                    label=f"{cond_labels.get(cond, cond)} {tgt}",
                    color=cond_colors.get(cond, "#636e72"),
                    marker=target_markers.get(tgt, "o"),
                    alpha=0.65,
                )
            if x_all.size >= 2 and np.nanmax(x_all) > np.nanmin(x_all):
                coeffs = np.polyfit(x_all, y_all, deg=1)
                line_x = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 50)
                line_y = np.polyval(coeffs, line_x)
                ax.plot(line_x, line_y, color=cond_colors.get(cond, "#636e72"), linestyle="-", linewidth=1.6, alpha=0.9)
                try:
                    stats_parts = []
                    for label, func in (("r", pearsonr), ("rho", spearmanr)):
                        if len(np.unique(x_all)) < 2 or len(np.unique(y_all)) < 2:
                            continue
                        with np.errstate(invalid="ignore"):
                            corr_val, p_val = func(x_all, y_all)
                        stats_parts.append(f"{label}={corr_val:.2f}, p={p_val:.3f}")
                    if stats_parts:
                        stat_lines.append(f"{cond_labels.get(cond, cond)} pooled: " + " | ".join(stats_parts))
                except Exception:
                    pass
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("LSTM test R^2")
        ax.set_title(f"{metric} vs R^2 (pooled A+B per condition)")
        if stat_lines:
            ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)

    for j in range(n_metrics, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Learning metrics vs pooled LSTM R^2 (pre/post)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    out_path = OUT_DIR / "LSTM_R2_vs_learning_metrics_pooled.png"
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved pooled-condition scatter grid to {out_path}")


def plot_scatter_by_metric_by_mouse(df: pd.DataFrame, metrics: list[str]):
    """Facet the main plot by mouse, keeping condition/target splits."""
    colors = {
        ("barrier_pre_flip", "A"): "#0984e3",
        ("barrier_pre_flip", "B"): "#6c5ce7",
        ("barrier_post_flip", "A"): "#d63031",
        ("barrier_post_flip", "B"): "#e17055",
    }
    markers = {"barrier_pre_flip": "o", "barrier_post_flip": "s"}
    cond_labels = {"barrier_pre_flip": "Pre flip", "barrier_post_flip": "Post flip"}
    mice = sorted(df["mouse"].unique())

    for metric in metrics:
        cols = 3
        rows = math.ceil(len(mice) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=False)

        for idx, mouse in enumerate(mice):
            ax = axes[idx // cols][idx % cols]
            stat_lines = []
            df_mouse = df[df["mouse"] == mouse]
            for cond in CONDITIONS:
                for target in sorted(df_mouse["target_label"].unique()):
                    mask = (df_mouse["condition"] == cond) & (df_mouse["target_label"] == target)
                    if not mask.any():
                        continue
                    x = df_mouse.loc[mask, metric].to_numpy(dtype=float)
                    y = df_mouse.loc[mask, "test_r2"].to_numpy(dtype=float)
                    finite = np.isfinite(x) & np.isfinite(y)
                    x = x[finite]
                    y = y[finite]
                    if x.size == 0:
                        continue
                    ax.scatter(
                        x,
                        y,
                        label=f"{cond_labels.get(cond, cond)} {target}",
                        marker=markers.get(cond, "o"),
                        color=colors.get((cond, target), "#636e72"),
                        alpha=0.75,
                    )
                    if x.size >= 2 and np.nanmax(x) > np.nanmin(x):
                        coeffs = np.polyfit(x, y, deg=1)
                        line_x = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                        line_y = np.polyval(coeffs, line_x)
                        ax.plot(line_x, line_y, color=colors.get((cond, target), "#636e72"), linestyle="-", linewidth=1.2, alpha=0.9)
                        try:
                            stats_parts = []
                            for label, func in (("r", pearsonr), ("rho", spearmanr)):
                                if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                                    continue
                                with np.errstate(invalid="ignore"):
                                    corr_val, p_val = func(x, y)
                                stats_parts.append(f"{label}={corr_val:.2f}, p={p_val:.3f}")
                            if stats_parts:
                                stat_lines.append(f"{cond_labels.get(cond, cond)} {target}: " + " | ".join(stats_parts))
                        except Exception:
                            pass
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel(metric.replace("_", " "))
            ax.set_ylabel("LSTM test R^2")
            ax.set_title(f"{mouse}")
            if stat_lines:
                ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)

        for j in range(len(mice), rows * cols):
            axes[j // cols][j % cols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{metric} vs LSTM R^2 by mouse (pre/post, split by target)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 0.88, 0.96))
        out_path = OUT_DIR / f"LSTM_R2_vs_learning_metrics_by_mouse_{metric}.png"
        fig.savefig(out_path, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"Saved by-mouse scatter grid for {metric} to {out_path}")


def plot_ab_diff_scatter(diff_df: pd.DataFrame, metrics: list[str]):
    if diff_df.empty:
        print("A-B diff scatter: no paired A/B sessions available; skipping.")
        return

    colors = {"barrier_pre_flip": "#6c5ce7", "barrier_post_flip": "#fd79a8"}
    markers = {"barrier_pre_flip": "o", "barrier_post_flip": "s"}
    cond_labels = {"barrier_pre_flip": "Pre flip", "barrier_post_flip": "Post flip"}

    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        stat_lines = []
        for cond in CONDITIONS:
            mask = diff_df["condition"] == cond
            if not mask.any():
                continue
            x = diff_df.loc[mask, metric].to_numpy(dtype=float)
            y = diff_df.loc[mask, "ab_diff"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if x.size == 0:
                continue
            ax.scatter(
                x,
                y,
                label=cond_labels.get(cond, cond),
                marker=markers.get(cond, "o"),
                color=colors.get(cond, "#636e72"),
                alpha=0.75,
            )
            if x.size >= 2 and np.nanmax(x) > np.nanmin(x):
                coeffs = np.polyfit(x, y, deg=1)
                line_x = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                line_y = np.polyval(coeffs, line_x)
                ax.plot(line_x, line_y, color=colors.get(cond, "#636e72"), linestyle="-", linewidth=1.2, alpha=0.9)
                try:
                    r_val, p_val = pearsonr(x, y)
                    stat_lines.append(f"{cond_labels.get(cond, cond)}: r={r_val:.2f}, p={p_val:.3f}")
                except Exception:
                    pass
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("A - B test R^2")
        ax.set_title(f"{metric} vs A-B R^2")
        if stat_lines:
            ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)

    for j in range(n_metrics, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Learning metrics vs LSTM (A-B) R^2 (pre/post flip)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    out_path = OUT_DIR / "LSTM_ABdiff_vs_learning_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved A-B diff scatter grid to {out_path}")


def _iqr_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask of values outside IQR fences."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr
    return (series < lower) | (series > upper)


def plot_scatter_by_metric_split_target(df: pd.DataFrame, metrics: list[str], filter_mode: str | None = None):
    """Scatter per metric with r values split by condition and target.

    filter_mode: None (all data),
                 "iqr_r2" / "iqr_metric" (keep IQR outliers),
                 "iqr_inlier_r2" / "iqr_inlier_metric" (keep data inside IQR fences).
    """
    colors = {
        ("barrier_pre_flip", "A"): "#0984e3",
        ("barrier_pre_flip", "B"): "#6c5ce7",
        ("barrier_post_flip", "A"): "#d63031",
        ("barrier_post_flip", "B"): "#e17055",
    }
    markers = {"A": "o", "B": "s"}
    cond_labels = {"barrier_pre_flip": "Pre flip", "barrier_post_flip": "Post flip"}

    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False, sharey=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        stat_lines = []
        for cond in CONDITIONS:
            for target in sorted(df["target_label"].unique()):
                mask = (df["condition"] == cond) & (df["target_label"] == target)
                if filter_mode == "iqr_r2" or filter_mode == "iqr_inlier_r2":
                    r2_series = df.loc[mask, "test_r2"]
                    if not r2_series.empty:
                        r2_outliers = _iqr_mask(r2_series)
                        keep_mask = r2_outliers if filter_mode == "iqr_r2" else ~r2_outliers
                        mask = mask & keep_mask.reindex(df.index, fill_value=False)
                    else:
                        mask = mask & False
                if filter_mode == "iqr_metric" or filter_mode == "iqr_inlier_metric":
                    cond_mask = df["condition"] == cond
                    metric_vals = df.loc[cond_mask, metric]
                    if not metric_vals.empty:
                        metric_outliers = _iqr_mask(metric_vals)
                        keep_mask = metric_outliers if filter_mode == "iqr_metric" else ~metric_outliers
                        mask = mask & keep_mask.reindex(df.index, fill_value=False)
                    else:
                        mask = mask & False
                if not mask.any():
                    continue
                x = df.loc[mask, metric].to_numpy(dtype=float)
                y = df.loc[mask, "test_r2"].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                x = x[finite]
                y = y[finite]
                if x.size == 0:
                    continue
                ax.scatter(
                    x,
                    y,
                    label=f"{cond_labels.get(cond, cond)} {target}",
                    marker=markers.get(target, "o"),
                    color=colors.get((cond, target), "#636e72"),
                    alpha=0.75,
                )
                if x.size >= 2 and np.nanmax(x) > np.nanmin(x):
                    coeffs = np.polyfit(x, y, deg=1)
                    line_x = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                    line_y = np.polyval(coeffs, line_x)
                    ax.plot(line_x, line_y, color=colors.get((cond, target), "#636e72"), linestyle="-", linewidth=1.2, alpha=0.9)
                    try:
                        stats_parts = []
                        for label, func in (("r", pearsonr), ("rho", spearmanr)):
                            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                                continue
                            with np.errstate(invalid="ignore"):
                                corr_val, p_val = func(x, y)
                            stats_parts.append(f"{label}={corr_val:.2f}, p={p_val:.3f}")
                        if stats_parts:
                            stat_lines.append(f"{cond_labels.get(cond, cond)} {target}: " + " | ".join(stats_parts))
                    except Exception:
                        pass
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("LSTM test R^2")
        title_suffix = ""
        if filter_mode == "iqr_r2":
            title_suffix = " (R^2 IQR outliers)"
        elif filter_mode == "iqr_metric":
            title_suffix = " (metric IQR outliers)"
        elif filter_mode == "iqr_inlier_r2":
            title_suffix = " (R^2 IQR inliers)"
        elif filter_mode == "iqr_inlier_metric":
            title_suffix = " (metric IQR inliers)"
        ax.set_title(f"{metric} vs R^2 (split by target){title_suffix}")
        if stat_lines:
            ax.text(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8)

    for j in range(n_metrics, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    title_suffix = ""
    out_name_suffix = "split_target"
    if filter_mode == "iqr_r2":
        title_suffix = " [R^2 IQR outliers]"
        out_name_suffix = "split_target_r2_iqr_outliers"
    elif filter_mode == "iqr_metric":
        title_suffix = " [metric IQR outliers]"
        out_name_suffix = "split_target_metric_iqr_outliers"
    elif filter_mode == "iqr_inlier_r2":
        title_suffix = " [R^2 IQR inliers]"
        out_name_suffix = "split_target_r2_iqr_inliers"
    elif filter_mode == "iqr_inlier_metric":
        title_suffix = " [metric IQR inliers]"
        out_name_suffix = "split_target_metric_iqr_inliers"
    fig.suptitle(f"Learning metrics vs LSTM R^2 (pre/post, split by target){title_suffix}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    out_path = OUT_DIR / f"LSTM_R2_vs_learning_metrics_{out_name_suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    print(f"Saved target-split scatter grid to {out_path}")


def main():
    merged_df, metrics = prepare_merged_df()
    if merged_df.empty:
        raise RuntimeError("Merged dataframe is empty; check inputs.")
    print(f"Merged {len(merged_df)} rows across {len(metrics)} metrics.")
    metrics_df = load_learning_metrics()
    lstm_df = load_lstm_df()
    diff_df = build_ab_diff_df(lstm_df, metrics_df, metrics)
    # plot_scatter_by_metric(merged_df, metrics)
    plot_scatter_by_metric_pooled(merged_df, metrics)
    # plot_scatter_by_metric_split_target(merged_df, metrics)
    # plot_scatter_by_metric_split_target(merged_df, metrics, filter_mode="iqr_r2")
    # plot_scatter_by_metric_split_target(merged_df, metrics, filter_mode="iqr_metric")
    # plot_scatter_by_metric_split_target(merged_df, metrics, filter_mode="iqr_inlier_r2")
    # plot_scatter_by_metric_split_target(merged_df, metrics, filter_mode="iqr_inlier_metric")
    # plot_scatter_by_metric_by_mouse(merged_df, metrics)
    plot_ab_diff_scatter(diff_df, metrics)


if __name__ == "__main__":
    main()
