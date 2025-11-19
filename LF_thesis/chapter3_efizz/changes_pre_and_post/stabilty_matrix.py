"""
Compute and plot a single pre→post transition matrix for tuning angles.

Logic
-----
1. Load the max-Rayleigh summary table for threat trials.
2. For each neuron/session, keep the row with the strongest Rayleigh magnitude
   in the pre and post barrier conditions.
3. Map all angle labels onto a small set of coarse categories.
4. Build a normalized pre→post transition matrix (row-wise probabilities) and
   display it as a heatmap with raw counts annotated in each cell.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- configuration ---
DATA_DIR = r"Z:\Jasmine_Laurence\rayleigh_analysis"
CSV_PATH = os.path.join(DATA_DIR, "threat_dict_max_rayleigh_flat.csv")
PRE_COND = "barrier_pre_flip"
POST_COND = "barrier_post_flip"

COARSE_ORDER = ["A", "B", "hdir", "hsa"]


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first column whose lower-case name matches a candidate."""
    name_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in name_map:
            return name_map[key]
    raise KeyError(f"Could not find any of {candidates} in dataframe columns.")


def normalize_angle_label(text: str) -> str:
    """Map a raw angle label to the coarse categories."""
    s = str(text).lower()
    if "postflip" in s:
        return "B"
    if "preflip" in s:
        return "A"
    if "hdir" in s:
        return "hdir"
    if "hsa" in s:
        return "hsa"
    return "other"


def build_top_pre_post(df: pd.DataFrame) -> pd.DataFrame:
    """Return merged dataframe containing top-pre and top-post rows per neuron."""
    session_col = find_column(df, ["session", "session_name", "sesh"])
    cluster_col = find_column(df, ["cluster_id", "cluster", "cell_id", "unit"])
    cond_col = find_column(df, ["condition"])
    compartment_col = find_column(df, ["compartment", "zone"])
    angle_col = find_column(df, ["angle", "angle_name", "angle_key"])
    rayleigh_col = next(
        c for c in df.columns if "rayleigh" in str(c).lower()
    )

    df_threat = df[
        (df[compartment_col].astype(str).str.lower() == "threat")
        & (df[cond_col].isin([PRE_COND, POST_COND]))
    ].copy()
    if df_threat.empty:
        raise SystemExit("No threat-compartment rows for the requested conditions.")

    idx = df_threat.groupby([session_col, cluster_col, cond_col])[rayleigh_col].idxmax()
    top_rows = df_threat.loc[idx].copy()

    pre = top_rows[top_rows[cond_col].eq(PRE_COND)].copy()
    post = top_rows[top_rows[cond_col].eq(POST_COND)].copy()

    merged = pre.merge(
        post, on=[session_col, cluster_col], suffixes=("_pre", "_post")
    )
    merged.rename(
        columns={
            f"{angle_col}_pre": "angle_pre",
            f"{angle_col}_post": "angle_post",
        },
        inplace=True,
    )
    return merged


def transition_matrix(pre_labels: np.ndarray, post_labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return probability transition matrix and raw counts."""
    pre = pd.Series(pre_labels, dtype=str).map(normalize_angle_label)
    post = pd.Series(post_labels, dtype=str).map(normalize_angle_label)
    cats = COARSE_ORDER + (["other"] if "other" in set(pre).union(post) else [])
    counts = (
        pd.crosstab(pre, post, dropna=False)
        .reindex(index=cats, columns=cats, fill_value=0)
    )
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return probs.round(3), counts


def plot_transition_heatmap(probs: pd.DataFrame, counts: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot the transition probability matrix with counts annotated."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True, dpi=140)
    sns.heatmap(
        probs,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#e6e6e6",
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Post angle", fontsize=13)
    ax.set_ylabel("Pre angle", fontsize=13)
    ax.set_title("Angle transition probabilities (all cells)", fontsize=15)
    ax.tick_params(axis="x", labelrotation=30, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    for i, row in enumerate(probs.index):
        for j, col in enumerate(probs.columns):
            count = int(counts.loc[row, col])
            if count > 0:
                ax.text(
                    j + 0.5,
                    i + 0.82,
                    f"n={count}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
    if save_path:
        fig.savefig(save_path, format="eps", dpi=150)
    plt.show()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    merged = build_top_pre_post(df)

    probs, counts = transition_matrix(
        merged["angle_pre"].to_numpy(),
        merged["angle_post"].to_numpy(),
    )
    save_path = r"Z:\Laurence\thesis\efizz_chapter\stability_matrix.eps"
    plot_transition_heatmap(probs, counts, save_path=save_path)


if __name__ == "__main__":
    main()
