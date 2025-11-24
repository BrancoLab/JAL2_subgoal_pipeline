import os

# --- configuration -----------------------------------------------------------
PICKLE_PATH = r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\A_vs_B_all_conditions_threat_zone.pkl"
SAVE_PATH   = r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\tuning_edge_percent_plot_all_conditions.eps"
TITLE_SUFFIX = "(Threat zone; TunED A vs B)"
CONDITIONS = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
A_KEYS = ["preflip_tuned", "h_preflipbar_a_tuned", "A_tuned"]
B_KEYS = ["postflip_tuned", "h_postflipbar_a_tuned", "B_tuned"]
SAVE_PATH_IQR = os.path.splitext(SAVE_PATH)[0] + "_iqr.eps"
USE_TTEST = True  # Set True to use paired t-test instead of Wilcoxon
# -----------------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import wilcoxon, ttest_rel


def load_final_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError("Loaded object is not a dict; verify the pickle.")
    return data


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
                else:  # shelter or other
                    open_n += int(a_flag)
                    closed_n += int(b_flag)

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


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    p = float(p)
    p = round(p, 2)
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


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
        stat, p = ttest_rel(a_vals, b_vals, nan_policy="omit")
    else:
        if np.allclose(a_vals, b_vals):
            return 0.0, 1.0, float(np.mean(b_vals - a_vals))
        stat, p = wilcoxon(a_vals, b_vals, alternative="two-sided", zero_method="wilcox", mode="auto")
    return stat, p, float(np.mean(b_vals - a_vals))


def paired_test_condition(df, condition, series_map=None, test_name="wilcoxon"):
    return paired_test_general(
        df,
        condition,
        "pct_A",
        condition,
        "pct_B",
        series_map=series_map,
        test_name=test_name,
    )


def plot_all_conditions(df, save_path=None, title_suffix="", use_iqr=False, test_name="wilcoxon"):
    if df.empty:
        raise ValueError("No data available to plot.")

    order = [
        ("shelter_only", "A | Shelter", "pct_A"),
        ("shelter_only", "B | Shelter", "pct_B"),
        ("barrier_pre_flip", "A | Pre-flip", "pct_A"),
        ("barrier_pre_flip", "B | Pre-flip", "pct_B"),
        ("barrier_post_flip", "A | Post-flip", "pct_A"),
        ("barrier_post_flip", "B | Post-flip", "pct_B"),
    ]

    cond_label_map = {
        "shelter_only": "Shelter only",
        "barrier_pre_flip": "Barrier pre-flip",
        "barrier_post_flip": "Barrier post-flip",
    }

    def metric_series(condition, metric):
        series = df[df["condition"] == condition].set_index("session")[metric].dropna()
        if use_iqr and len(series) >= 4:
            q1, q3 = series.quantile([0.25, 0.75])
            series = series[(series >= q1) & (series <= q3)]
        return series

    extra_keys = {
        ("barrier_pre_flip", "pct_open_over_total"),
        ("barrier_post_flip", "pct_closed_over_total"),
        ("barrier_pre_flip", "pct_closed_over_total"),
        ("barrier_post_flip", "pct_open_over_total"),
    }
    metric_keys = {(cond, metric) for cond, _, metric in order}
    metric_keys |= extra_keys
    series_map = {(cond, metric): metric_series(cond, metric) for cond, metric in metric_keys}

    means = []
    xticklabels = []
    all_values = []
    for cond, label, metric in order:
        vals = series_map[(cond, metric)]
        means.append(float(vals.mean()) if len(vals) else 0.0)
        xticklabels.append(label)

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#6c5ce7", "#74b9ff"] * 3
    ax.bar(x, means, color=bar_colors, alpha=0.85)

    rng = np.random.default_rng(7)
    sessions = df["session"].unique()
    for session in sessions:
        vals = []
        for cond, label, metric in order:
            series = series_map[(cond, metric)]
            value = float(series.get(session)) if session in series.index else np.nan
            vals.append(value)
            if np.isfinite(value):
                all_values.append(value)
        xs = x + rng.normal(0, 0.02, size=len(order))
        ax.scatter(xs, vals, marker="x", color="gray", alpha=0.6, s=50)
        for idx in range(0, len(order), 2):
            a, b = vals[idx], vals[idx + 1]
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([xs[idx], xs[idx + 1]], [a, b], color="gray", alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=15, ha="right")
    ax.set_ylabel("Fraction of cells (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    data_max = max(all_values) if all_values else (max(means) if means else 0.02)
    ylim_top = data_max * 1.2 if data_max > 0 else 0.05
    ax.set_ylim(0, max(0.03, ylim_top))
    ax.grid(axis="y", alpha=0.3)
    suffix = f"{title_suffix} (IQR)" if use_iqr else title_suffix
    ax.set_title(f"TunED A vs B fractions across conditions\n{suffix}", fontsize=14)

    comparisons_info = []
    for cond in CONDITIONS:
        stat, p, mean_diff = paired_test_condition(
            df,
            cond,
            series_map=series_map,
            test_name=test_name,
        )
        comparisons_info.append((cond, stat, p, mean_diff))
        tag = " [IQR]" if use_iqr else ""
        print(
            f"{cond_label_map[cond]} (A vs B){tag} [{test_name}]: "
            f"stat={stat:.4f}, p={p:.4g}, mean(B-A)={mean_diff:.4f}"
        )

    pad = ax.get_ylim()[1] * 0.03
    max_ann = 0.0

    def add_sig_bar(x1, x2, base_height, p_value, pad_scale=1.0):
        nonlocal max_ann
        label = stars(p_value)
        line_height = base_height + pad * pad_scale
        ax.plot(
            [x1, x1, x2, x2],
            [line_height, line_height + pad * 0.5, line_height + pad * 0.5, line_height],
            lw=1.3,
            color="k",
        )
        ax.text((x1 + x2) / 2, line_height + pad * 0.55, label, ha="center", va="bottom", fontsize=11)
        max_ann = max(max_ann, line_height + pad * 0.6)

    # Within-condition A vs B
    for idx, cond in enumerate(CONDITIONS):
        base = idx * 2
        top = max(means[base], means[base + 1])
        _, _, p, _ = comparisons_info[idx]
        add_sig_bar(base, base + 1, top, p, pad_scale=1.1)

    # Cross-condition comparisons
    cross_tests = [
        ("A Shelter vs A Pre", 0, 2, ("shelter_only", "pct_A"), ("barrier_pre_flip", "pct_A")),
        ("A Shelter vs A Post", 0, 4, ("shelter_only", "pct_A"), ("barrier_post_flip", "pct_A")),
        ("A Pre vs A Post", 2, 4, ("barrier_pre_flip", "pct_open_over_total"), ("barrier_post_flip", "pct_closed_over_total")),
        ("B Shelter vs B Pre", 1, 3, ("shelter_only", "pct_B"), ("barrier_pre_flip", "pct_B")),
        ("B Shelter vs B Post", 1, 5, ("shelter_only", "pct_B"), ("barrier_post_flip", "pct_B")),
        ("B Pre vs B Post", 3, 5, ("barrier_pre_flip", "pct_closed_over_total"), ("barrier_post_flip", "pct_open_over_total")),
    ]

    for idx, (name, start_idx, end_idx, cfg_a, cfg_b) in enumerate(cross_tests):
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
        print(f"{name}{tag} [{test_name}]: stat={stat:.4f}, p={p_val:.4g}, mean diff={mean_diff:.4f}")
        top = max(means[start_idx], means[end_idx])
        add_sig_bar(start_idx, end_idx, top + pad * (idx + 1.5), p_val, pad_scale=0.6)

    # ensure axes cover annotations
    if max_ann:
        ax.set_ylim(0, max(ax.get_ylim()[1], max_ann + pad))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format="eps")
        print(f"Saved figure to {save_path}")
    plt.show()

    results_summary = pd.DataFrame(
        [
            dict(
                condition=cond_label_map[c],
                test_stat=stat,
                p_value=p,
                mean_B_minus_A=md,
            )
            for c, stat, p, md in comparisons_info
        ]
    )
    return results_summary


# ---- run everything ---------------------------------------------------------
final_results = load_final_results(PICKLE_PATH)
session_df = aggregate_sessions(final_results, CONDITIONS)
test_name = "ttest" if USE_TTEST else "wilcoxon"
stats_summary = plot_all_conditions(
    session_df,
    save_path=SAVE_PATH,
    title_suffix=TITLE_SUFFIX,
    use_iqr=False,
    test_name=test_name,
)
print("\nSession table (first 5 rows):")
print(session_df.head().to_string(index=False))
print("\nPaired-test summary (full data):")
print(stats_summary.to_string(index=False))

# stats_summary_iqr = plot_all_conditions(
#     session_df,
#     save_path=SAVE_PATH_IQR,
#     title_suffix=TITLE_SUFFIX,
#     use_iqr=True,
#     test_name=test_name,
# )
# print("\nPaired-test summary (IQR-filtered):")
# print(stats_summary_iqr.to_string(index=False))
