"""
Identify two classes of neurons for manual inspection and quantify whether they
show statistically meaningful increases in tuning when the barrier flips:

1) Pre-silent-B:
   - Pre angle already equals B (h_postflipbar_a).
   - Pre firing rate is low and Rayleigh magnitude is high, but both increase
     further after the flip (post Rayleigh > pre Rayleigh, post FR > pre FR).
   - Post angle remains B with sufficient Rayleigh support.

2) Switch-to-B:
   - Pre angle does not equal B, yet the unit shows low FR + high Rayleigh in pre.
   - After the flip it becomes tuned to B (and meets the strict Rayleigh cut).
   - As with the first class, Rayleigh and firing rate must increase post flip.

For each class, the script records session/unit IDs and computes the mean
increase in Rayleigh and firing rate (Δpost - Δpre). A two-sided permutation
test is applied: the observed class mean is compared to a null distribution of
means obtained by repeatedly sampling equally sized groups from all cells that
show positive increases. p-values are converted to significance stars for plotting.
"""

import ast
import os
import re
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- user config ------------------------------------------------------------
CSV_PATH = r"Z:\Jasmine_Laurence\rayleigh_analysis\threat_dict_max_rayleigh_flat.csv"
OUTPUT_CSV = r"Z:\Laurence\thesis\efizz_chapter\candidate_cells.csv"

POST_LABEL = "h_postflipbar_a"   # canonical "B" label
LOW_FR_THRESHOLD = 5.0           # Hz; adjust as needed
HIGH_RAYLEIGH_THRESHOLD = 0.35   # Rayleigh magnitude; adjust as needed
RAYLEIGH_STRICT_CUT = 0.25       # same cut used in modeling for "strict" cells
N_PERM = 2000
RNG_SEED = 0
FIG_DIR = r"Z:\Laurence\thesis\efizz_chapter"
# ---------------------------------------------------------------------------


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lookup = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f"None of the columns {candidates} were found.")


def normalize_angle_label(label: str) -> str:
    s = str(label).lower()
    if "postflip" in s:
        return "h_postflipbar_a"
    if "preflip" in s:
        return "h_preflipbar_a"
    if "hdir" in s:
        return "hdir"
    if "hsa" in s:
        return "hsa"
    return s


def to_float_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []
        try:
            arr = np.asarray(ast.literal_eval(s), dtype=float)
        except Exception:
            parts = [p for p in re.split(r"[,;\s]+", s.strip("[]")) if p]
            arr = np.asarray([float(p) for p in parts], dtype=float) if parts else np.array([])
    else:
        return []
    arr = np.ravel(arr)
    arr = arr[np.isfinite(arr)]
    return arr.tolist()


def fr_stats(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=np.nan, frac_zero=np.nan, max=np.nan)
    mean = float(np.nanmean(arr))
    frac_zero = float(np.mean(arr <= 1e-12))
    return dict(mean=mean, frac_zero=frac_zero, max=float(np.nanmax(arr)))


def _find_rayleigh_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "rayleigh" in str(col).lower():
            return col
    raise KeyError("Could not locate a Rayleigh column.")


def permutation_means(values: np.ndarray, sample_size: int, n_perm: int, rng: np.random.Generator):
    """Return permutation distribution of mean values (without replacement)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if sample_size <= 0:
        return np.array([])
    if sample_size > values.size:
        raise ValueError("Sample size larger than permutation pool.")
    samples = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        idx = rng.choice(values.size, size=sample_size, replace=False)
        samples[i] = np.mean(values[idx])
    return samples


def p_value_two_sided(observed: float, null_distribution: np.ndarray) -> float:
    if null_distribution.size == 0 or not np.isfinite(observed):
        return np.nan
    more = np.mean(null_distribution >= observed)
    less = np.mean(null_distribution <= observed)
    p = 2 * min(more, less)
    return float(min(1.0, p))


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_patterns_combined(pattern_results,
                           pool_df: pd.DataFrame,
                           out_path: Optional[str] = None,
                           scatter_sample: int = 300):
    """Create a combined scatter + marginal hist plot for all patterns."""
    import matplotlib.gridspec as gridspec

    if not pattern_results:
        return

    pool_color = "#95a5a6"
    scatter_sample = min(scatter_sample, len(pool_df))
    pool_sample = pool_df.sample(scatter_sample, random_state=RNG_SEED) if scatter_sample > 0 else pool_df

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=(4, 1.5), height_ratios=(1.5, 4), hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Null histograms
    ax_histx.hist(pool_df["delta_ray"], bins=40, color=pool_color, alpha=0.4, label="Permutation pool")
    ax_histy.hist(pool_df["delta_fr"], bins=40, orientation="horizontal", color=pool_color, alpha=0.4)

    legend_entries = [("Permutation pool", pool_color)]
    summary_lines = []
    for res in pattern_results:
        data = res["subset"]
        color = res["color"]
        label = res["label"]
        ax_histx.hist(data["delta_ray"], bins=max(10, len(data)//2), color=color, alpha=0.7)
        ax_histy.hist(data["delta_fr"], bins=max(10, len(data)//2), orientation="horizontal", color=color, alpha=0.7)
        ax_scatter.scatter(data["delta_ray"], data["delta_fr"], color=color, edgecolor="k", s=45, alpha=0.85)
        legend_entries.append((label, color))
        summary_lines.append(
            f"{label}: Δfr={res['stats']['delta_fr_mean']:.2f} ({res['stats']['fr_stars']}), "
            f"Δray={res['stats']['delta_ray_mean']:.3f} ({res['stats']['ray_stars']})"
        )

    # pool scatter
    ax_scatter.scatter(pool_sample["delta_ray"], pool_sample["delta_fr"], color=pool_color, alpha=0.3, s=20)
    ax_scatter.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_scatter.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax_scatter.set_xlabel("Δ Rayleigh (post - pre)")
    ax_scatter.set_ylabel("Δ Mean firing rate (post - pre) [Hz]")

    handles = []
    labels = []
    for name, color in legend_entries:
        h = ax_scatter.scatter([], [], color=color, label=name)
        handles.append(h)
        labels.append(name)
    ax_scatter.legend(handles, labels, loc="lower right", frameon=False)

    annotation = "\n".join(summary_lines)
    ax_scatter.text(0.02, 0.98, annotation, ha="left", va="top", transform=ax_scatter.transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=9)
    ax_histx.set_ylabel("Count")
    ax_histy.set_xlabel("Count")
    fig.suptitle("Δ metrics for candidate classes vs permutation pool", fontsize=14)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved combined plot to {out_path}")
    plt.show()


def build_summary_table(df: pd.DataFrame, pre_cond="barrier_pre_flip", post_cond="barrier_post_flip"):
    session_col = find_col(df, ["session", "session_name", "sesh"])
    cluster_col = find_col(df, ["cluster_id", "cluster", "cell_id", "unit", "neuron"])
    condition_col = find_col(df, ["condition"])
    compartment_col = find_col(df, ["compartment", "zone"])
    angle_col = find_col(df, ["angle", "angle_name", "angle_key"])
    rayleigh_col = _find_rayleigh_col(df)
    fr_col = "firing_rate_hz"

    threat = df[
        (df[compartment_col].astype(str).str.lower() == "threat")
        & (df[condition_col].isin([pre_cond, post_cond]))
    ].copy()
    if threat.empty:
        raise SystemExit("No THREAT rows for target conditions.")

    idx = threat.groupby([session_col, cluster_col, condition_col])[rayleigh_col].idxmax()
    top = threat.loc[idx].copy()

    pre = top[top[condition_col].eq(pre_cond)].copy()
    post = top[top[condition_col].eq(post_cond)].copy()
    merged = pre.merge(post, on=[session_col, cluster_col], suffixes=("_pre", "_post"))

    pre_vecs = merged[f"{fr_col}_pre"].apply(to_float_list)
    post_vecs = merged[f"{fr_col}_post"].apply(to_float_list)
    stats_pre = pre_vecs.apply(fr_stats).apply(pd.Series)
    stats_post = post_vecs.apply(fr_stats).apply(pd.Series)

    summary = pd.DataFrame(
        {
            "session": merged[session_col].values,
            "cluster_id": merged[cluster_col].values,
            "pre_angle": merged[f"{angle_col}_pre"].map(normalize_angle_label).values,
            "post_angle": merged[f"{angle_col}_post"].map(normalize_angle_label).values,
            "rayleigh_pre": merged[f"{rayleigh_col}_pre"].astype(float).values,
            "rayleigh_post": merged[f"{rayleigh_col}_post"].astype(float).values,
            "fr_mean_pre": stats_pre["mean"].values,
            "fr_mean_post": stats_post["mean"].values,
            "fr_fraczero_pre": stats_pre["frac_zero"].values,
            "fr_max_pre": stats_pre["max"].values,
            "fr_max_post": stats_post["max"].values,
        }
    )

    summary["Y_post_is_postflip_strict"] = (
        (summary["post_angle"] == POST_LABEL) & (summary["rayleigh_post"] > RAYLEIGH_STRICT_CUT)
    ).astype(int)

    return summary


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Could not find source CSV: {CSV_PATH}")

    df_raw = pd.read_csv(CSV_PATH)
    summary = build_summary_table(df_raw)
    summary["delta_fr"] = summary["fr_mean_post"] - summary["fr_mean_pre"]
    summary["delta_ray"] = summary["rayleigh_post"] - summary["rayleigh_pre"]

    condition_post_B = summary["post_angle"].eq(POST_LABEL) & summary["Y_post_is_postflip_strict"].eq(1)
    rayleigh_increase = summary["rayleigh_post"] > summary["rayleigh_pre"]
    firing_increase = summary["fr_mean_post"] > summary["fr_mean_pre"]

    silently_B = (
        condition_post_B
        & summary["pre_angle"].eq(POST_LABEL)
        & summary["fr_mean_pre"].le(LOW_FR_THRESHOLD)
        & summary["rayleigh_pre"].ge(HIGH_RAYLEIGH_THRESHOLD)
        & rayleigh_increase
        & firing_increase
    )

    switch_to_B = (
        condition_post_B
        & summary["pre_angle"].ne(POST_LABEL)
        & summary["fr_mean_pre"].le(LOW_FR_THRESHOLD)
        & summary["rayleigh_pre"].ge(HIGH_RAYLEIGH_THRESHOLD)
        & rayleigh_increase
        & firing_increase
    )

    summary.loc[silently_B, "pattern"] = "pre-silent-B"
    summary.loc[switch_to_B, "pattern"] = "switch-to-B"

    candidates = summary.loc[summary["pattern"].notna()].copy()
    columns_to_keep = [
        "session",
        "cluster_id",
        "pre_angle",
        "post_angle",
        "fr_mean_pre",
        "fr_mean_post",
        "fr_max_pre",
        "fr_max_post",
        "rayleigh_pre",
        "rayleigh_post",
        "pattern",
    ]
    candidates = candidates[columns_to_keep].sort_values(["pattern", "session", "cluster_id"]).reset_index(drop=True)

    print(f"Found {len(candidates)} candidate cells "
          f"(silent-B: {silently_B.sum()}, switch-to-B: {switch_to_B.sum()}).")

    if OUTPUT_CSV:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        candidates.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved candidate list to {OUTPUT_CSV}")

    print(candidates)

    # ---- Pattern-level stats and permutation plots ----
    pool_mask = rayleigh_increase & firing_increase
    pool_df = summary.loc[pool_mask].dropna(subset=["delta_fr", "delta_ray"]).copy()
    if pool_df.empty:
        raise SystemExit("Permutation pool is empty after requiring positive increases.")
    pattern_meta = {
        "pre-silent-B": {"label": "Pre-silent B", "color": "#d35400"},
        "switch-to-B": {"label": "Switch to B", "color": "#2980b9"},
    }

    pattern_results = []
    for idx, (pattern_key, meta) in enumerate(pattern_meta.items()):
        subset = summary.loc[summary["pattern"] == pattern_key].dropna(subset=["delta_fr", "delta_ray"])
        if subset.empty:
            print(f"No cells found for {meta['label']}; skipping permutation analysis.")
            continue
        n_cells = len(subset)
        print(f"\nPermutation analysis for {meta['label']} (n={n_cells})")

        rng_fr = np.random.default_rng(RNG_SEED + idx * 2)
        rng_ray = np.random.default_rng(RNG_SEED + idx * 2 + 1)

        fr_perm = permutation_means(pool_df["delta_fr"].values, n_cells, N_PERM, rng_fr)
        ray_perm = permutation_means(pool_df["delta_ray"].values, n_cells, N_PERM, rng_ray)

        fr_mean = float(subset["delta_fr"].mean())
        ray_mean = float(subset["delta_ray"].mean())

        fr_p = p_value_two_sided(fr_mean, fr_perm)
        ray_p = p_value_two_sided(ray_mean, ray_perm)

        print(f"  Δfr mean={fr_mean:.3f}, p={fr_p:.4f} (two-sided permutation)")
        print(f"  Δray mean={ray_mean:.3f}, p={ray_p:.4f} (two-sided permutation)")

        stats = {
            "delta_fr_mean": fr_mean,
            "delta_ray_mean": ray_mean,
            "fr_p": fr_p,
            "ray_p": ray_p,
            "fr_stars": p_to_stars(fr_p),
            "ray_stars": p_to_stars(ray_p),
            "color": meta["color"],
        }
        pattern_results.append(
            {
                "key": pattern_key,
                "label": meta["label"],
                "subset": subset,
                "stats": stats,
                "color": meta["color"],
            }
        )

    if pattern_results:
        fig_path = None
        if FIG_DIR:
            os.makedirs(FIG_DIR, exist_ok=True)
            fig_path = os.path.join(FIG_DIR, "combined_delta_permutation.eps")
        plot_patterns_combined(pattern_results, pool_df, out_path=fig_path)


if __name__ == "__main__":
    main()
