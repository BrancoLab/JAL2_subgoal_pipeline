import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_rel, wilcoxon

CONDITIONS = ["barrier_pre_flip", "barrier_post_flip"]
SAVE_ROOT = Path(r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED")
PICKLE_AB = SAVE_ROOT / "A_vs_B_all_conditions_threat_zone.pkl"
PICKLE_A_HSA = SAVE_ROOT / "A_vs_HSA.pkl"
PICKLE_B_HSA = SAVE_ROOT / "B_vs_HSA.pkl"
SAVE_PATH = SAVE_ROOT / "tuning_edge_percent_only_pre_post.eps"
TITLE_SUFFIX = "(Threat zone; TunED A-only vs B-only)"
A_KEYS = ["preflip_tuned", "h_preflipbar_a_tuned", "A_tuned"]
B_KEYS = ["postflip_tuned", "h_postflipbar_a_tuned", "B_tuned"]


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    p = float(p)
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def _flag_value(flags, candidates):
    for key in candidates:
        if key in flags:
            return bool(flags.get(key, False))
    return False


def aggregate_sessions(final_results, conditions):
    rows = []
    for session, by_condition in final_results.items():
        for cond in conditions:
            clusters = by_condition.get(cond, {}) or {}
            if not isinstance(clusters, dict):
                continue
            total = len(clusters)
            if total == 0:
                continue

            a_n = b_n = mixed_n = 0
            open_n = closed_n = 0
            for flags in clusters.values():
                a_flag = _flag_value(flags, A_KEYS)
                b_flag = _flag_value(flags, B_KEYS)
                a_n += int(a_flag)
                b_n += int(b_flag)
                mixed_n += int(bool(flags.get("mixed_tuning", False)))
                if cond == "barrier_pre_flip":
                    open_n += int(a_flag)
                    closed_n += int(b_flag)
                elif cond == "barrier_post_flip":
                    open_n += int(b_flag)
                    closed_n += int(a_flag)

            rows.append(
                dict(
                    session=session,
                    condition=cond,
                    total_cells=total,
                    pct_A=a_n / total,
                    pct_B=b_n / total,
                    pct_mixed=mixed_n / total,
                    pct_open_over_total=open_n / total,
                    pct_closed_over_total=closed_n / total,
                )
            )
    return pd.DataFrame(rows)


def paired_test_general(df, cond_a, metric_a, cond_b, metric_b, *, series_map=None, test_name="wilcoxon"):
    if series_map is not None:
        a_series = series_map.get((cond_a, metric_a), pd.Series(dtype=float))
        b_series = series_map.get((cond_b, metric_b), pd.Series(dtype=float))
    else:
        a_series = df[df["condition"] == cond_a].set_index("session")[metric_a].dropna()
        b_series = df[df["condition"] == cond_b].set_index("session")[metric_b].dropna()
    common = a_series.index.intersection(b_series.index)
    min_pairs = 2 if test_name == "ttest" else 3
    if len(common) < min_pairs:
        return np.nan, np.nan, np.nan
    a_vals = a_series.loc[common].values
    b_vals = b_series.loc[common].values
    if test_name == "ttest":
        stat, p_val = ttest_rel(a_vals, b_vals, nan_policy="omit")
    else:
        if np.allclose(a_vals, b_vals):
            return 0.0, 1.0, float(np.mean(b_vals - a_vals))
        stat, p_val = wilcoxon(a_vals, b_vals, alternative="two-sided", zero_method="wilcox", mode="auto")
    return stat, p_val, float(np.mean(b_vals - a_vals))


def paired_test_condition(df, condition, *, series_map=None, test_name="wilcoxon"):
    return paired_test_general(
        df,
        condition,
        "pct_A",
        condition,
        "pct_B",
        series_map=series_map,
        test_name=test_name,
    )


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected pickle structure in {path}")
    return data


def allowed_ids_from_hsa(results, tuning_key):
    """Return ids that are not HSA-only (allow mixed cells)."""
    allowed = {}
    for session, cond_dict in results.items():
        allowed_session = {}
        for cond, clusters in (cond_dict or {}).items():
            allowed_ids = set()
            if isinstance(clusters, dict):
                for cluster_id, flags in clusters.items():
                    mixed = bool(flags.get("mixed_tuning", False))
                    target_tuned = bool(flags.get(tuning_key, False))
                    hsa_tuned = bool(flags.get("hsa_tuned", False))
                    if mixed or target_tuned or not hsa_tuned:
                        allowed_ids.add(cluster_id)
            allowed_session[cond] = allowed_ids
        allowed[session] = allowed_session
    return allowed


def build_filtered_results(results_ab, allowed_pre, allowed_post):
    filtered = {}
    for session, cond_dict in results_ab.items():
        filtered_session = {}
        for cond in CONDITIONS:
            clusters_ab = cond_dict.get(cond, {}) or {}
            session_cond = {}
            pre_allowed_ids = allowed_pre.get(session, {}).get(cond, set())
            post_allowed_ids = allowed_post.get(session, {}).get(cond, set())

            for cluster_id, ab_flags in clusters_ab.items():
                pre_tuned = bool(ab_flags.get("h_preflipbar_a_tuned", False)) and (cluster_id in pre_allowed_ids)
                post_tuned = bool(ab_flags.get("h_postflipbar_a_tuned", False)) and (cluster_id in post_allowed_ids)
                pre_only = pre_tuned and not post_tuned
                post_only = post_tuned and not pre_tuned
                session_cond[cluster_id] = {
                    "h_preflipbar_a_tuned": pre_only,
                    "h_postflipbar_a_tuned": post_only,
                    "mixed_tuning": False,
                }
            filtered_session[cond] = session_cond
        filtered[session] = filtered_session
    return filtered


def metric_series(df, condition, metric, *, use_iqr=False):
    series = df[df["condition"] == condition].set_index("session")[metric].dropna()
    if use_iqr and len(series) >= 4:
        q1, q3 = series.quantile([0.25, 0.75])
        series = series[(series >= q1) & (series <= q3)]
    return series


def plot_pre_post_only(df, *, save_path=None, title_suffix="", use_iqr=False, test_name="wilcoxon"):
    if df.empty:
        raise ValueError("No rows available for plotting.")

    order = [
        ("barrier_pre_flip", "A | Pre-flip", "pct_A"),
        ("barrier_pre_flip", "B | Pre-flip", "pct_B"),
        ("barrier_post_flip", "A | Post-flip", "pct_A"),
        ("barrier_post_flip", "B | Post-flip", "pct_B"),
    ]

    metric_keys = {(cond, metric) for cond, _, metric in order}
    metric_keys |= {
        ("barrier_pre_flip", "pct_open_over_total"),
        ("barrier_pre_flip", "pct_closed_over_total"),
        ("barrier_post_flip", "pct_open_over_total"),
        ("barrier_post_flip", "pct_closed_over_total"),
    }
    series_map = {(cond, metric): metric_series(df, cond, metric, use_iqr=use_iqr) for cond, metric in metric_keys}

    means = [float(series_map[(cond, metric)].mean()) if len(series_map[(cond, metric)]) else 0.0 for cond, _, metric in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ["#6c5ce7", "#74b9ff", "#6c5ce7", "#74b9ff"]
    ax.bar(x, means, color=bar_colors, alpha=0.85)

    rng = np.random.default_rng(12)
    sessions = df["session"].unique()
    all_values = []
    for session in sessions:
        vals = []
        jitter = x + rng.normal(0, 0.02, size=len(order))
        for idx, (cond, _, metric) in enumerate(order):
            series = series_map[(cond, metric)]
            value = float(series.get(session)) if session in series.index else np.nan
            vals.append(value)
            if np.isfinite(value):
                all_values.append(value)
                ax.scatter(jitter[idx], value, marker="x", color="gray", alpha=0.6, s=45)
        for pair_start in range(0, len(order), 2):
            a, b = vals[pair_start], vals[pair_start + 1]
            if np.isfinite(a) and np.isfinite(b):
                ax.plot(
                    [jitter[pair_start], jitter[pair_start + 1]],
                    [a, b],
                    color="gray",
                    alpha=0.4,
                    linewidth=1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in order], rotation=15, ha="right")
    ax.set_ylabel("Fraction of cells (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ylim_top = max(all_values) if all_values else max(means)
    ax.set_ylim(0, max(0.03, ylim_top * 1.25 if ylim_top else 0.05))
    ax.grid(axis="y", alpha=0.3)
    suffix = f"{title_suffix} (IQR)" if use_iqr else title_suffix
    ax.set_title(f"Exclusive TunED A vs B per condition\n{suffix}", fontsize=14)

    pad = ax.get_ylim()[1] * 0.03
    max_ann = 0.0

    def add_sig_bar(x1, x2, base_height, p_value, pad_scale=1.0):
        nonlocal max_ann
        label = stars(p_value)
        line_height = base_height + pad * pad_scale
        ax.plot(
            [x1, x1, x2, x2],
            [line_height, line_height + pad * 0.6, line_height + pad * 0.6, line_height],
            lw=1.1,
            color="k",
        )
        ax.text(
            (x1 + x2) / 2,
            line_height + pad * 0.65,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
        )
        max_ann = max(max_ann, line_height + pad * 0.7)

    comparisons_info = []
    conditions_to_use = CONDITIONS
    for cond in conditions_to_use:
        stat, p_val, mean_diff = paired_test_condition(
            df,
            cond,
            series_map=series_map,
            test_name=test_name,
        )
        comparisons_info.append((cond, stat, p_val, mean_diff))
        tag = " [IQR]" if use_iqr else ""
        label = "Pre-flip" if cond == "barrier_pre_flip" else "Post-flip"
        print(
            f"{label} (A vs B){tag} [{test_name}]: "
            f"stat={stat:.4f}, p={p_val:.4g}, mean(B-A)={mean_diff:.4f}"
        )

    for idx, cond in enumerate(conditions_to_use):
        base = idx * 2
        top = max(means[base], means[base + 1])
        _, _, p_val, _ = comparisons_info[idx]
        if np.isfinite(p_val) and stars(p_val) != "ns":
            add_sig_bar(base, base + 1, top, p_val, pad_scale=1.1)

    cross_tests = [
        ("A Pre vs A Post", 0, 2, ("barrier_pre_flip", "pct_open_over_total"), ("barrier_post_flip", "pct_closed_over_total")),
        ("B Pre vs B Post", 1, 3, ("barrier_pre_flip", "pct_closed_over_total"), ("barrier_post_flip", "pct_open_over_total")),
    ]
    for idx, (label, start_idx, end_idx, cfg_a, cfg_b) in enumerate(cross_tests, start=1):
        stat, p_val, mean_diff = paired_test_general(
            df,
            cfg_a[0],
            cfg_a[1],
            cfg_b[0],
            cfg_b[1],
            series_map=series_map,
            test_name=test_name,
        )
        tag = " [IQR]" if use_iqr else ""
        print(f"{label}{tag} [{test_name}]: stat={stat:.4f}, p={p_val:.4g}, mean diff={mean_diff:.4f}")
        top = max(means[start_idx], means[end_idx])
        if np.isfinite(p_val) and stars(p_val) != "ns":
            add_sig_bar(start_idx, end_idx, top + pad * (idx + 0.4), p_val, pad_scale=0.6)

    if max_ann:
        ax.set_ylim(0, max(ax.get_ylim()[1], max_ann + pad))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="eps")
        print(f"Saved figure to {save_path}")
    plt.show()

    summary = pd.DataFrame(
        [
            dict(condition=("Pre-flip" if cond == "barrier_pre_flip" else "Post-flip"), test_stat=stat, p_value=p_val, mean_B_minus_A=mean_diff)
            for cond, stat, p_val, mean_diff in comparisons_info
        ]
    )
    return summary


def main():
    print("Loading TunED comparison pickles...")
    results_ab = load_pickle(PICKLE_AB)
    results_a_hsa = load_pickle(PICKLE_A_HSA)
    results_b_hsa = load_pickle(PICKLE_B_HSA)

    allowed_pre = allowed_ids_from_hsa(results_a_hsa, "h_preflipbar_a_tuned")
    allowed_post = allowed_ids_from_hsa(results_b_hsa, "h_postflipbar_a_tuned")
    filtered_results = build_filtered_results(results_ab, allowed_pre, allowed_post)
    session_df = aggregate_sessions(filtered_results, CONDITIONS)
    if session_df.empty:
        raise RuntimeError("No sessions survived exclusivity filtering.")

    stats_summary = plot_pre_post_only(
        session_df,
        save_path=SAVE_PATH,
        title_suffix=TITLE_SUFFIX,
        use_iqr=False,
        test_name="wilcoxon",
    )
    print("\nSession table (first 5 rows):")
    print(session_df.head().to_string(index=False))
    print("\nPaired-test summary:")
    print(stats_summary.to_string(index=False))


if __name__ == "__main__":
    main()
