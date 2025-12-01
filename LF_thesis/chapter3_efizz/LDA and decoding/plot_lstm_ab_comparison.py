"""Plot paired R^2 comparisons (A vs B) for LSTM decoding per condition."""

from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

RESULTS_PKL = Path(r"Z:\Laurence\thesis\efizz_chapter\LSTM_results.pkl")
OUT_DIR = Path(r"Z:\Laurence\thesis\efizz_chapter") / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MAP = {"h_preflipbar_a": "A", "h_postflipbar_a": "B"}
TEST_METHOD = "ttest"  # "ttest" or "wilcoxon"
MOUSE_LINE_TEST = "wilcoxon"  # "ttest" or "wilcoxon"

MICE_GROUPS = {
    "JAL6": ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may"],
    "JAL4": ["JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"],
    "JAL5": ["JAL005_8thSept", "JAL005_21stSept"],
}


def load_df() -> pd.DataFrame:
    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)
    df = pd.DataFrame(results, columns=["session", "condition", "target", "test_r2"])
    df = df[df["target"].isin(TARGET_MAP)].copy()
    df["target_label"] = df["target"].map(TARGET_MAP)
    return df


def build_pivot(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    df_cond = df[df["condition"] == condition]
    pivot = df_cond.pivot_table(index="session", columns="target_label", values="test_r2")
    return pivot.dropna()


def prepare_pivots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    piv_pre = build_pivot(df, "barrier_pre_flip")
    piv_post = build_pivot(df, "barrier_post_flip")
    return piv_pre, piv_post


def paired_test(a: pd.Series, b: pd.Series):
    if TEST_METHOD == "wilcoxon":
        return wilcoxon(a, b)
    return ttest_rel(a, b)


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bar(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10)


def plot_session_group(
    piv_pre: pd.DataFrame, piv_post: pd.DataFrame, sessions, title: str, filename: str
):
    available_sessions = set(piv_pre.index) & set(piv_post.index)
    valid_sessions = sorted(available_sessions & set(sessions))
    if not valid_sessions:
        print(f"{title}: no sessions with paired A/B data; skipping plot.")
        return

    piv_pre = piv_pre.loc[valid_sessions]
    piv_post = piv_post.loc[valid_sessions]
    fig, ax = plt.subplots(figsize=(9, 6))
    for sess in valid_sessions:
        ax.plot(
            [0, 1],
            [piv_pre.loc[sess, "A"], piv_pre.loc[sess, "B"]],
            marker="o",
            color="#74b9ff",
            alpha=0.6,
        )
        ax.plot(
            [2, 3],
            [piv_post.loc[sess, "A"], piv_post.loc[sess, "B"]],
            marker="o",
            color="#ff7675",
            alpha=0.6,
        )

    means = [
        piv_pre["A"].mean(),
        piv_pre["B"].mean(),
        piv_post["A"].mean(),
        piv_post["B"].mean(),
    ]

    ax.bar([0, 1, 2, 3], means, color=["#0984e3", "#74b9ff", "#d63031", "#ff7675"], alpha=0.4)
    for x, m in zip([0, 1, 2, 3], means):
        ax.text(x, m + 0.02, f"{m:.3f}", ha="center")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["Pre flip A", "Pre flip B", "Post flip A", "Post flip B"],
        rotation=10,
        ha="right",
    )
    ax.set_ylabel("Test R^2")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)

    comparisons = [
        {"xs": (0, 1), "series": (piv_pre["A"], piv_pre["B"]), "label": "Pre A vs B"},
        {"xs": (2, 3), "series": (piv_post["A"], piv_post["B"]), "label": "Post A vs B"},
        {"xs": (0, 2), "series": (piv_pre["A"], piv_post["A"]), "label": "A pre vs post"},
        {"xs": (1, 3), "series": (piv_pre["B"], piv_post["B"]), "label": "B pre vs post"},
    ]

    data_min = min(piv_pre.to_numpy().min(), piv_post.to_numpy().min())
    data_max = max(piv_pre.to_numpy().max(), piv_post.to_numpy().max())
    margin = max(0.05, (data_max - data_min) * 0.1 if data_max != data_min else 0.1)
    bottom = data_min - margin
    top = data_max + margin

    summary_lines = []
    if len(valid_sessions) >= 2:
        y_range = data_max - data_min if data_max != data_min else 0.1
        y_step = max(0.04, y_range * 0.08)
        bar_h = y_step * 0.6
        start_y = data_max + y_step
        for idx, comp in enumerate(comparisons):
            _, p_val = paired_test(*comp["series"])
            stars = p_to_stars(p_val)
            y = start_y + idx * (y_step + bar_h)
            add_sig_bar(ax, comp["xs"][0], comp["xs"][1], y, bar_h, stars)
            summary_lines.append(f"{comp['label']}: p={p_val:.4f} ({stars})")
        top = max(top, start_y + len(comparisons) * (y_step + bar_h) + y_step)
        print(f"{title} (n={len(valid_sessions)}):\n" + "\n".join(summary_lines))
    else:
        print(f"{title}: not enough sessions for paired stats (n={len(valid_sessions)}).")

    ax.set_ylim(bottom, top)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.show()
    plt.close(fig)


def plot_combined(df: pd.DataFrame, piv_pre: pd.DataFrame | None = None, piv_post: pd.DataFrame | None = None):
    if piv_pre is None or piv_post is None:
        piv_pre, piv_post = prepare_pivots(df)
    sessions = set(piv_pre.index) & set(piv_post.index)
    plot_session_group(
        piv_pre,
        piv_post,
        sessions,
        title="LSTM A/B decoding across conditions",
        filename="LSTM_pre_post_AB.png",
    )


def plot_by_mouse(piv_pre: pd.DataFrame, piv_post: pd.DataFrame):
    for mouse, sessions in MICE_GROUPS.items():
        plot_session_group(
            piv_pre,
            piv_post,
            sessions,
            title=f"{mouse} LSTM A/B decoding across conditions",
            filename=f"LSTM_pre_post_AB_{mouse}.png",
        )


def plot_ab_differences(piv_pre: pd.DataFrame, piv_post: pd.DataFrame):
    """Plot A-B difference per session for pre/post flip and show the mean."""
    sessions = sorted(set(piv_pre.index) & set(piv_post.index))
    if not sessions:
        print("A-B difference plot: no overlapping sessions; skipping.")
        return

    diff_pre = piv_pre.loc[sessions, "A"] - piv_pre.loc[sessions, "B"]
    diff_post = piv_post.loc[sessions, "A"] - piv_post.loc[sessions, "B"]
    means = {"Pre flip": diff_pre.mean(), "Post flip": diff_post.mean()}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    for ax, diff, label, color in [
        (axes[0], diff_pre, "Pre flip A-B", "#0984e3"),
        (axes[1], diff_post, "Post flip A-B", "#d63031"),
    ]:
        mean_val = means["Pre flip"] if "Pre" in label else means["Post flip"]
        ax.bar(range(len(sessions)), diff, color=color, alpha=0.7)
        ax.axhline(mean_val, color="k", linestyle="--", linewidth=1)
        ax.text(
            len(sessions) - 0.5,
            mean_val + 0.01,
            f"mean={mean_val:.3f}",
            ha="right",
            va="bottom",
            fontsize=10,
        )
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(sessions, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("A - B (Test R^2)")
        ax.set_title(label)
        ax.axhline(0, color="gray", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "LSTM_AB_difference_pre_post.png", dpi=200)
    plt.show()
    plt.close(fig)
    print(
        "A-B difference means:\n"
        f"Pre flip mean: {means['Pre flip']:.4f}\n"
        f"Post flip mean: {means['Post flip']:.4f}"
    )


def plot_mouse_mean_lines(piv_pre: pd.DataFrame, piv_post: pd.DataFrame):
    """Average A-B per mouse for pre/post and draw one line per mouse plus overall means."""
    pre_diff = (piv_pre["A"] - piv_pre["B"]).dropna()
    post_diff = (piv_post["A"] - piv_post["B"]).dropna()

    session_to_mouse = {sess: mouse for mouse, sess_list in MICE_GROUPS.items() for sess in sess_list}
    mouse_colors = {
        "JAL6": "#00b894",
        "JAL3": "#0984e3",
        "JAL7": "#6c5ce7",
        "JAL8": "#e84393",
        "JAL4": "#f9ca24",
        "JAL5": "#e17055",
    }

    data = []
    for mouse in MICE_GROUPS:
        pre_vals = pre_diff[[s for s in pre_diff.index if session_to_mouse.get(s) == mouse]]
        post_vals = post_diff[[s for s in post_diff.index if session_to_mouse.get(s) == mouse]]
        pre_mean = pre_vals.mean() if len(pre_vals) else float("nan")
        post_mean = post_vals.mean() if len(post_vals) else float("nan")
        data.append({"mouse": mouse, "pre": pre_mean, "post": post_mean})
    mouse_df = pd.DataFrame(data)

    if mouse_df[["pre", "post"]].isna().all().all():
        print("Mouse mean plot: no data available; skipping.")
        return

    overall_means = {"pre": pre_diff.mean() if len(pre_diff) else float("nan"), "post": post_diff.mean() if len(post_diff) else float("nan")}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([0, 1], [overall_means["pre"], overall_means["post"]], color=["#b2bec3", "#636e72"], alpha=0.5, width=0.6, label="Overall mean")
    for x, label in zip([0, 1], ["pre", "post"]):
        mean_val = overall_means[label]
        if pd.notna(mean_val):
            ax.text(x, mean_val + 0.005, f"{mean_val:.3f}", ha="center", fontsize=10)

    for _, row in mouse_df.iterrows():
        mouse = row["mouse"]
        color = mouse_colors.get(mouse, "#636e72")
        xs = []
        ys = []
        if pd.notna(row["pre"]):
            xs.append(0)
            ys.append(row["pre"])
        if pd.notna(row["post"]):
            xs.append(1)
            ys.append(row["post"])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.2, alpha=0.9, label=mouse)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre flip A-B", "Post flip A-B"])
    ax.set_ylabel("A - B (Test R^2)")
    ax.set_title("A-B R^2 per mouse (mean across sessions)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate legend entries
    seen = {}
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen[l] = True
        new_handles.append(h)
        new_labels.append(l)
    ax.legend(new_handles, new_labels, loc="best")

    paired = mouse_df.dropna(subset=["pre", "post"])
    if len(paired) >= 2:
        if MOUSE_LINE_TEST == "ttest":
            _, p_val = ttest_rel(paired["pre"], paired["post"])
        else:
            _, p_val = wilcoxon(paired["pre"], paired["post"])
        stars = p_to_stars(p_val)
        top_y = ax.get_ylim()[1]
        y = max(top_y, float(np.nanmax(mouse_df[["pre", "post"]].values)) + 0.02)
        add_sig_bar(ax, 0, 1, y, 0.02, stars)
        ax.set_ylim(ax.get_ylim()[0], y + 0.06)
        method_label = "paired t-test" if MOUSE_LINE_TEST == "ttest" else "paired Wilcoxon"
        print(f"Mouse-level {method_label} pre vs post: p={p_val:.4f} ({stars}), n={len(paired)}")
    else:
        print(f"Mouse-level test skipped (paired n={len(paired)}).")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "LSTM_AB_mouse_mean_lines.eps", dpi=200)
    plt.show()
    plt.close(fig)


def main():
    df = load_df()
    piv_pre, piv_post = prepare_pivots(df)
    plot_combined(df, piv_pre, piv_post)
    # plot_by_mouse(piv_pre, piv_post)
    plot_ab_differences(piv_pre, piv_post)
    plot_mouse_mean_lines(piv_pre, piv_post)


if __name__ == "__main__":
    main()
