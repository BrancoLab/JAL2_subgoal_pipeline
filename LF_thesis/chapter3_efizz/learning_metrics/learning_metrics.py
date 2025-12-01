# Learning metrics to extract per session, linked to the conditions of escapes and homings
# Number of homings - how many homings occured per session in each condition
# Spatial efficiency of escapes - how direct were the escapes in each condition - e.g we could get the average of all escapes in each condition
# Spatial efficiency of homings - how direct were the homings in each condition - e.g we could get the average of all homings in each condition
# Average speed of escapes and homings in each condition
# Rate of homings across some time window - e.g number of homings in each condition per 10 minutes

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behave_analysis.process.session import get_experiment
from loguru import logger

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr, JAL7_30apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may

experiments_objects = [
    JAL6_flip7_1apr,
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_28mar,
    JAL3_25aug,
    JAL3_1sept,
    JAL3_4sept,
    JAL3_7sept,
    JAL005_8thSept,
    JAL005_21stSept,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip5_22mar,
    JAL7_flip2_12mar,
    JAL7_23apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_14may,
    JAL4_3rdSept,
    JAL4_19thSept,
    JAL4_28aug,
    JAL4_11thSept,
]

session_NAMES = [
    "JAL6_flip7_1apr",
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL005_8thSept",
    "JAL005_21stSept",
    "JAL7_sesh8_9apr",
    "JAL7_sesh9_16apr",
    "JAL7_flip5_22mar",
    "JAL7_flip2_12mar",
    "JAL7_23apr",
    "JAL8_flip1_25apr",
    "JAL8_flip2_29apr",
    "JAL8_flip4_10may",
    "JAL8_14may",
    "JAL4_3rdSept",
    "JAL4_19thSept",
    "JAL4_28aug",
    "JAL4_11thSept",
]


# Function to load homings object
def load_homings_object(session_path):
    """Load homings object from a session"""
    homings_path = os.path.join(session_path, "homings", "homings_obj.pkl")

    if os.path.exists(homings_path):
        try:
            with open(homings_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading homings object from {session_path}: {e}")

    return None


def load_escapes_object(session_path):
    """Load escapes object from a session, checking multiple possible paths"""
    possible_paths = [
        os.path.join(session_path, "escapes", "escapes_obj.pkl"),
        os.path.join(session_path, "escape", "escapes_obj.pkl"),
        os.path.join(session_path, "escapes_obj.pkl"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading escapes object from {path}: {e}")

    return None


# Function to count homings and escapes

from collections import defaultdict
dictionary_results = defaultdict(dict)

conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]

for i, session in enumerate(experiments_objects):
    session_name = session_NAMES[i]
    logger.info(f"Processing session: {session_name}")
    loaded_session = get_experiment(session)
    session_path = os.path.join(loaded_session.base_path, loaded_session.processed_path)

    # Get homings object and metrics
    homings_obj = load_homings_object(session_path)
    if homings_obj is None:
        print(f"No homings object found for session {session_NAMES[i]}. Skipping.")
        continue
    number_of_homings = len(homings_obj.onset_frames)  # scalar
    avg_speed_homings = homings_obj.avg_speed  # (, len) - float
    spatial_efficiency_homings = homings_obj.spatial_efficiency  # (, len) - float
    conditions_homings = homings_obj.homing_condition  # (, len) - string
    timings_homings = [x / 40 for x in homings_obj.onset_frames]  # (, len) - float

    # Get escapes object and metrics
    escapes_obj = load_escapes_object(session_path)
    if escapes_obj is None:
        print(f"No escapes object found for session {session_NAMES[i]}. Skipping.")
        continue
    number_of_escapes = len(escapes_obj.stim_onset_frames)  # scalar
    escape_speeds = escapes_obj.avg_speed  # (, len) - float
    spatial_efficiency_escapes = escapes_obj.spatial_efficiency  # (, len) - float
    conditions_escapes = escapes_obj.escape_condition  # (, len) - string
    timings_escapes = [x / 40 for x in escapes_obj.stim_onset_frames]  # (, len) - float

    # Store results in dictionary
    dictionary_results[session_name]["number_of_homings"] = number_of_homings # a scalar number of homings
    dictionary_results[session_name]["avg_speed_homings"] = avg_speed_homings # a list of average speeds for each homing
    dictionary_results[session_name]["spatial_efficiency_homings"] = spatial_efficiency_homings # a list of spatial efficiencies for each homing
    dictionary_results[session_name]["conditions_homings"] = conditions_homings # a list of conditions for each homing (e.g "shelter_only", "barrier_pre_flip", "barrier_post_flip")
    dictionary_results[session_name]["timings_homings"] = timings_homings # a list o ftimings in seconds when each homing started
    
    dictionary_results[session_name]["number_of_escapes"] = number_of_escapes # a scalar number of escapes
    dictionary_results[session_name]["escape_speeds"] = escape_speeds # a list of average speeds for each escape
    dictionary_results[session_name]["spatial_efficiency_escapes"] = spatial_efficiency_escapes # a list of spatial efficiencies for each escape
    dictionary_results[session_name]["conditions_escapes"] = conditions_escapes # a list of conditions for each escape (e.g "shelter_only", "barrier_pre_flip", "barrier_post_flip")
    dictionary_results[session_name]["timings_escapes"] = timings_escapes # a list of timings in seconds when each escape started


def _rate_per_ten_minutes(times: np.ndarray) -> float:
    if len(times) == 0:
        return 0.0
    span = float(np.max(times) - np.min(times))
    span = max(span, 600.0)
    return float(len(times) / (span / 600.0))


def _cv(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    mean_val = float(np.mean(values))
    if np.isclose(mean_val, 0):
        return np.nan
    return float(np.std(values, ddof=0) / mean_val)


def build_condition_rows(results_dict):
    rows = []
    for session, metrics in results_dict.items():
        if not metrics:
            continue
        homing_speeds = np.asarray(metrics.get("avg_speed_homings", []), dtype=float)
        homing_eff = np.asarray(metrics.get("spatial_efficiency_homings", []), dtype=float)
        homing_conditions = np.asarray(metrics.get("conditions_homings", []), dtype=object)
        homing_times = np.asarray(metrics.get("timings_homings", []), dtype=float)

        escape_speeds = np.asarray(metrics.get("escape_speeds", []), dtype=float)
        escape_eff = np.asarray(metrics.get("spatial_efficiency_escapes", []), dtype=float)
        escape_conditions = np.asarray(metrics.get("conditions_escapes", []), dtype=object)
        escape_times = np.asarray(metrics.get("timings_escapes", []), dtype=float)

        for cond in conditions:
            homing_mask = homing_conditions == cond
            escape_mask = escape_conditions == cond

            homing_count = int(np.sum(homing_mask))
            escape_count = int(np.sum(escape_mask))

            homing_speed_vals = homing_speeds[homing_mask]
            homing_eff_vals = homing_eff[homing_mask]
            homing_time_vals = homing_times[homing_mask]

            escape_speed_vals = escape_speeds[escape_mask]
            escape_eff_vals = escape_eff[escape_mask]
            escape_time_vals = escape_times[escape_mask]

            inter_homing = np.diff(np.sort(homing_time_vals)) if homing_count > 1 else np.array([])
            inter_escape = np.diff(np.sort(escape_time_vals)) if escape_count > 1 else np.array([])

            rows.append(
                dict(
                    session=session,
                    condition=cond,
                    homings_count=homing_count,
                    escapes_count=escape_count,
                    homings_mean_speed=float(np.mean(homing_speed_vals)) if homing_count else np.nan,
                    escapes_mean_speed=float(np.mean(escape_speed_vals)) if escape_count else np.nan,
                    homings_mean_efficiency=float(np.mean(homing_eff_vals)) if homing_count else np.nan,
                    escapes_mean_efficiency=float(np.mean(escape_eff_vals)) if escape_count else np.nan,
                    homings_speed_cv=_cv(homing_speed_vals),
                    escapes_speed_cv=_cv(escape_speed_vals),
                    homings_efficiency_std=float(np.std(homing_eff_vals, ddof=0)) if homing_count else np.nan,
                    escapes_efficiency_std=float(np.std(escape_eff_vals, ddof=0)) if escape_count else np.nan,
                    homings_rate_per_10min=_rate_per_ten_minutes(homing_time_vals),
                    escapes_rate_per_10min=_rate_per_ten_minutes(escape_time_vals),
                    homings_median_latency=float(np.median(homing_time_vals)) if homing_count else np.nan,
                    escapes_median_latency=float(np.median(escape_time_vals)) if escape_count else np.nan,
                    mean_escape_times=float(np.mean(escape_time_vals)) if escape_count else np.nan,
                    median_escape_times=float(np.median(escape_time_vals)) if escape_count else np.nan,
                    homings_iei_mean=float(np.mean(inter_homing)) if len(inter_homing) else np.nan,
                    escapes_iei_mean=float(np.mean(inter_escape)) if len(inter_escape) else np.nan,
                )
            )
    return pd.DataFrame(rows)


if not dictionary_results:
    logger.warning("No learning metrics were collected; skipping aggregation.")
else:
    OUTPUT_DIR = Path(r"Z:\Laurence\thesis\efizz_chapter") / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    METRICS_CSV = OUTPUT_DIR / "learning_metrics_per_condition.csv"
    METRICS_PKL = OUTPUT_DIR / "learning_metrics_per_condition.pkl"
    PLOT_PATH = OUTPUT_DIR / "learning_metrics_summary.png"

    metrics_df = build_condition_rows(dictionary_results)
    if metrics_df.empty:
        logger.warning("Metrics dataframe is empty after processing.")
    else:
        metrics_df["homing_to_escape_ratio"] = metrics_df["homings_count"] / metrics_df["escapes_count"].replace(0, np.nan)

        bootstrap_metrics = [
            "homings_count",
            "escapes_count",
            "homings_mean_speed",
            "escapes_mean_speed",
            "homings_mean_efficiency",
            "escapes_mean_efficiency",
            "homings_rate_per_10min",
            "escapes_rate_per_10min",
        ]

        BOOTSTRAP_ITER = 5000
        rng = np.random.default_rng(2025)

        def bootstrap_pvalue(values, observed):
            values = values[np.isfinite(values)]
            if len(values) < 2 or not np.isfinite(observed):
                return np.nan
            samples = rng.choice(values, size=BOOTSTRAP_ITER, replace=True)
            ge = np.mean(samples >= observed)
            le = np.mean(samples <= observed)
            p = 2 * min(ge, le)
            if p == 0:
                p = 1.0 / BOOTSTRAP_ITER
            return float(min(p, 1.0))

        for metric in bootstrap_metrics:
            pvals = []
            mean_diff = metrics_df[metric] - metrics_df.groupby("condition")[metric].transform("mean")
            metrics_df[f"{metric}_diff_from_mean"] = mean_diff
            for _, row in metrics_df.iterrows():
                cond_vals = metrics_df[metrics_df["condition"] == row["condition"]][metric].to_numpy(dtype=float)
                pvals.append(bootstrap_pvalue(cond_vals, row[metric]))
            p_col = f"{metric}_pval"
            metrics_df[p_col] = pvals
            metrics_df[f"{metric}_significant"] = metrics_df[p_col] < 0.05

        metrics_df.to_csv(METRICS_CSV, index=False)
        metrics_df.to_pickle(METRICS_PKL)
        logger.info(f"Saved per-condition learning metrics to {METRICS_CSV}")

        plot_metrics = [
            ("homings_count", "Homing count"),
            ("escapes_count", "Escape count"),
            ("homings_mean_speed", "Mean homing speed"),
            ("escapes_mean_speed", "Mean escape speed"),
            ("homings_mean_efficiency", "Mean homing efficiency"),
            ("escapes_mean_efficiency", "Mean escape efficiency"),
            ("homings_rate_per_10min", "Homing rate /10 min"),
            ("escapes_rate_per_10min", "Escape rate /10 min"),
            ("homing_to_escape_ratio", "Homing/Escape ratio"),
        ]

        n_cols = 3
        n_rows = int(np.ceil(len(plot_metrics) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False)
        axes = axes.flatten()
        rng = np.random.default_rng(42)

        def plot_metric(ax, metric, title):
            bar_means = []
            for cond_idx, cond in enumerate(conditions):
                cond_values = metrics_df[metrics_df["condition"] == cond][metric].dropna()
                bar_means.append(cond_values.mean() if not cond_values.empty else 0.0)
                if not cond_values.empty:
                    jitter = rng.normal(0, 0.04, size=len(cond_values))
                    ax.scatter(
                        np.full(len(cond_values), cond_idx) + jitter,
                        cond_values,
                        color="gray",
                        alpha=0.6,
                        s=25,
                    )
            ax.bar(range(len(conditions)), bar_means, color=["#6c5ce7", "#74b9ff", "#a29bfe"], alpha=0.45)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=20, ha="right")
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=0.3)

        for idx, (metric, title) in enumerate(plot_metrics):
            plot_metric(axes[idx], metric, title)

        for idx in range(len(plot_metrics), len(axes)):
            axes[idx].axis("off")

        fig.suptitle("Learning metrics per session and condition", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
        logger.info(f"Saved summary plot to {PLOT_PATH}")

        pre_post_conditions = {"barrier_pre_flip", "barrier_post_flip"}
        significant_rows = []
        for metric in bootstrap_metrics:
            sig_col = f"{metric}_significant"
            p_col = f"{metric}_pval"
            diff_col = f"{metric}_diff_from_mean"
            flagged = metrics_df[
                metrics_df["condition"].isin(pre_post_conditions)
                & metrics_df[sig_col]
                & metrics_df[metric].notna()
            ]
            for _, row in flagged.iterrows():
                direction = "higher" if row[diff_col] >= 0 else "lower"
                significant_rows.append(
                    dict(
                        session=row["session"],
                        condition=row["condition"],
                        metric=metric,
                        value=row[metric],
                        p_value=row[p_col],
                        direction=direction,
                    )
                )
        if significant_rows:
            print("\nSessions with bootstrap-significant deviations (p < 0.05) in pre/post conditions:")
            for entry in significant_rows:
                print(
                    f"- {entry['session']} [{entry['condition']}]: "
                    f"{entry['metric']}={entry['value']:.3f} (p={entry['p_value']:.4f}, {entry['direction']})"
                )


# turn the csv intoa  dataframe
df = pd.read_csv(METRICS_CSV)
print("\nFull learning metrics dataframe:")

# filter to only preflip condition
preflip_df = df[df["condition"] == "barrier_pre_flip"]

print(preflip_df.columns.to_list())

# filter the bottom three homing effiency sessions with the lowest average homing efficiency in pre-flip condition
bottom_sessions_homing_pre = preflip_df.nsmallest(5, "homings_mean_efficiency")["session"].to_list()
print(bottom_sessions_homing_pre)

# print the session names of the smallest escape efficiency in pre-flip condition
bottom_escape_sessions = preflip_df.nsmallest(5, "escapes_mean_efficiency")["session"].to_list()
print(bottom_escape_sessions)

bottom_homing_counts = preflip_df.nsmallest(5, "homings_count")["session"].to_list()
print(bottom_homing_counts)

bottom_pre_intersection = set(bottom_sessions_homing_pre).intersection(set(bottom_escape_sessions))
print("Intersection of lowest homing efficiencies, lowest escape efficiencies, and lowest homing counts pre-flip:")
print(bottom_pre_intersection)

postflip_df = df[df["condition"] == "barrier_post_flip"]

# filter the top three homing efficiency sessions with the highest average homing efficiency in post-flip condition
top_escape_sessions = postflip_df.nlargest(5, "escapes_mean_efficiency")["session"].to_list()
print("Top sessions with highest escape efficiency post-flip:")
print(top_escape_sessions)

top_counts_homings = postflip_df.nlargest(5, "homings_count")["session"].to_list()
print("Top sessions with highest homing counts post-flip:")
print(top_counts_homings)

fastest_homing_speeds = postflip_df.nlargest(5, "homings_mean_speed")["session"].to_list()
print("Top sessions with fastest homing speeds post-flip:")
print(fastest_homing_speeds)

top_efficiency_homings = postflip_df.nlargest(5, "homings_mean_efficiency")["session"].to_list()
print("Top sessions with highest homing efficiencies post-flip:")
print(top_efficiency_homings)

intersection_sessions = set(top_escape_sessions).intersection(set(top_counts_homings)).intersection(set(top_efficiency_homings))
print("Intersection of fastest homing speeds, top homing efficiencies, and top homing counts:")
print(intersection_sessions)

# look for the smallset in post flip condition
lowest_escape_efficiency = postflip_df.nsmallest(5, "escapes_mean_efficiency")["session"].to_list()
print("Sessions with lowest escape efficiency post-flip:")
print(lowest_escape_efficiency)
lowest_homing_efficiency = postflip_df.nsmallest(5, "homings_mean_efficiency")["session"].to_list()
print("Sessions with lowest homing efficiency post-flip:")
print(lowest_homing_efficiency)
lowest_homing_counts = postflip_df.nsmallest(5, "homings_count")["session"].to_list()
print("Sessions with lowest homing counts post-flip:")
print(lowest_homing_counts) 

low_post_flip_intersection = set(lowest_escape_efficiency).intersection(set(lowest_homing_efficiency)).intersection(set(lowest_homing_counts))
print("Intersection of lowest escape efficiency, lowest homing efficiency, and lowest homing counts post-flip:")
print(low_post_flip_intersection)

