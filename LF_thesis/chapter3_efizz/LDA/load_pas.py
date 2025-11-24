import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept, JAL3_22aug
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_11thSept, JAL4_28aug
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept, JAL005_5thSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_flip3_18mar, JAL6_flip7_1apr, JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip3_7may, JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip4_10may, JAL8_14may

from behave_analysis.process.session import get_experiment

# Sessions to process
EXPERIMENTS = [
    JAL3_7sept, JAL3_4sept, JAL3_1sept, JAL3_25aug, JAL3_22aug,
    JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept,
    JAL005_8thSept, JAL005_21stSept, JAL005_5thSept,
    JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr,
    JAL7_sesh8_9apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_sesh9_16apr, JAL7_23apr,
    JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip3_7may, JAL8_flip4_10may, JAL8_14may,
]

SESSION_NAMES = [
    "JAL3_7sept", "JAL3_4sept", "JAL3_1sept", "JAL3_25aug", "JAL3_22aug",
    "JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept",
    "JAL005_8thSept", "JAL005_21stSept", "JAL005_5thSept",
    "JAL6_28mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_flip3_18mar", "JAL6_flip7_1apr",
    "JAL7_sesh8_9apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_sesh9_16apr", "JAL7_23apr",
    "JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip3_7may", "JAL8_flip4_10may", "JAL8_14may",
]

# Mice groups based on session names
mice_groups = {
    "JAL6": ['JAL6_flip7_1apr', 'JAL6_flip3_18mar', 'JAL6_flip4_21mar', 'JAL6_flip5_25mar', 'JAL6_28mar'],
    "JAL3": ['JAL3_25aug', 'JAL3_1sept', 'JAL3_4sept', 'JAL3_7sept'],
    "JAL7": ['JAL7_sesh8_9apr', 'JAL7_sesh9_16apr', 'JAL7_flip5_22mar', 'JAL7_flip2_12mar', 'JAL7_23apr'],
    "JAL8": ['JAL8_flip1_25apr', 'JAL8_flip2_29apr', 'JAL8_flip4_10may', 'JAL8_14may'],
    "JAL4": ['JAL4_3rdSept', 'JAL4_19thSept', 'JAL4_28aug', 'JAL4_11thSept'],
    "JAL5": ['JAL005_8thSept', 'JAL005_21stSept']}

CONDITIONS = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
TIME_WINDOWS = ["second_half"]  # adjust to whatever you actually ran
SETTINGS = "all_angles_fr_excl_prox_15cm"
OUTPUT_PATH = r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\all_prediction_accuracy.pkl"
DELTA_PLOT_PATH = OUTPUT_PATH.replace(".pkl", "_delta_plot.png")


def load_prediction_accuracy(session, settings, cond, time_cond):
    """Load a single PA pickle, skipping missing / inaccessible files."""
    rel_path = os.path.join(
        session.base_path,
        session.processed_path,
        "models",
        "LDA",
        settings,
        "good",
        time_cond,
        "all",
        cond,
        f"good_{cond}_LDA_pa.pkl",
    )

    if not os.path.exists(rel_path):
        print(f"[SKIP] Missing file: {rel_path}")
        return None

    try:
        with open(rel_path, "rb") as fh:
            return pickle.load(fh)
    except PermissionError:
        print(f"[SKIP] Permission denied: {rel_path}")
        return None
    except Exception as exc:
        print(f"[SKIP] Failed to load {rel_path}: {exc}")
        return None


def collect_prediction_accuracy():
    results = defaultdict(dict)

    for sesh, sesh_name in zip(EXPERIMENTS, SESSION_NAMES):
        session = get_experiment(sesh)
        print(f"\n=== {sesh_name} ===")

        for t_cond in TIME_WINDOWS:
            for cond in CONDITIONS:
                coef = load_prediction_accuracy(session, SETTINGS, cond, t_cond)
                if coef is None:
                    continue
                results[sesh_name].setdefault(t_cond, {})[cond] = coef

    return results


def compute_delta(entry):
    if not entry:
        return np.nan
    pre = entry.get("h_preflipbar_a")
    post = entry.get("h_postflipbar_a")
    if pre is None or post is None:
        return np.nan
    return pre - post


def find_sessions_pre_higher_post_lower(pa_dict, time_window):
    matching = []
    for session_name, time_dict in pa_dict.items():
        cond_data = time_dict.get(time_window)
        if not cond_data:
            continue
        shelter = compute_delta(cond_data.get("shelter_only"))
        pre = compute_delta(cond_data.get("barrier_pre_flip"))
        post = compute_delta(cond_data.get("barrier_post_flip"))
        if np.isfinite(shelter) and np.isfinite(pre) and np.isfinite(post):
            if pre > shelter and post < shelter:
                matching.append(session_name)
    return matching


def plot_delta_by_session(pa_dict, time_window, conditions, save_path=None):
    if not pa_dict:
        print("No prediction accuracy entries to plot.")
        return

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(pa_dict)))

    for color, (session_name, time_dict) in zip(colors, pa_dict.items()):
        cond_data = time_dict.get(time_window)
        if not cond_data:
            continue

        deltas = []
        for cond in conditions:
            entry = cond_data.get(cond)
            deltas.append(compute_delta(entry))

        if not np.any(np.isfinite(deltas)):
            continue

        ax.plot(
            x,
            deltas,
            marker="o",
            linewidth=1.5,
            color=color,
            label=session_name,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Δ decoding accuracy (A - B)")
    ax.set_title(f"Δ(A - B) decoding accuracy per session [{time_window}]")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize="x-small",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_delta_by_mouse(pa_dict, time_window, conditions, save_path=None):
    if not pa_dict:
        print("No prediction accuracy entries to plot.")
        return

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(mice_groups)))

    for color, (mouse, session_list) in zip(colors, mice_groups.items()):
        mouse_sessions = {s: pa_dict[s] for s in session_list if s in pa_dict}
        if not mouse_sessions:
            continue

        all_deltas = []
        for _, time_dict in mouse_sessions.items():
            cond_data = time_dict.get(time_window)
            if not cond_data:
                continue
            deltas = [compute_delta(cond_data.get(cond)) for cond in conditions]
            if np.any(np.isfinite(deltas)):
                all_deltas.append(deltas)

        if not all_deltas:
            continue

        mean_deltas = np.nanmean(np.array(all_deltas), axis=0)
        ax.plot(
            x,
            mean_deltas,
            marker="o",
            linewidth=2.0,
            color=color,
            label=mouse,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Δ decoding accuracy (A - B)")
    ax.set_title(f"Mean Δ(A - B) decoding accuracy per mouse [{time_window}]")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize="small")
    fig.tight_layout()
    if save_path:
        path = save_path.replace(".png", "_per_mouse.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    all_pa = collect_prediction_accuracy()
    if not all_pa:
        print("No prediction accuracy files were loaded.")
        return

    time_window = TIME_WINDOWS[0]
    plot_delta_by_session(all_pa, time_window, CONDITIONS, save_path=DELTA_PLOT_PATH)
    plot_delta_by_mouse(all_pa, time_window, CONDITIONS, save_path=DELTA_PLOT_PATH)
    matching = find_sessions_pre_higher_post_lower(all_pa, time_window)
    if matching:
        print("\nSessions with barrier_pre > shelter and barrier_post < shelter (ΔA-B):")
        for name in matching:
            print(" -", name)
    else:
        print("\nNo sessions met the pre>shelter and post<shelter Δ criteria.")

if __name__ == "__main__":
    main()
