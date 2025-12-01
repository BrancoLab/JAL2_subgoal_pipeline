"""Event-level learning metrics with regression and classification."""

from __future__ import annotations

import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_11thSept, JAL4_19thSept, JAL4_28aug, JAL4_3rdSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_21stSept, JAL005_8thSept
from behave_analysis.database.Experiments.JAL006_ex import (
    JAL6_28mar,
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_flip7_1apr,
)
from behave_analysis.database.Experiments.JAL007_ex import (
    JAL7_23apr,
    JAL7_30apr,
    JAL7_flip2_12mar,
    JAL7_flip5_22mar,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
)
from behave_analysis.database.Experiments.JAL008_ex import (
    JAL8_14may,
    JAL8_21may,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_tiny_3may,
)
from behave_analysis.process.session import get_experiment
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
REBUILD_EVENT_DATA = False
OUTPUT_DIR = Path(r"Z:\Laurence\thesis\efizz_chapter") / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_CSV = OUTPUT_DIR / "event_learning_metrics.csv"

LEARNING_METRICS_PATH = Path(r"Z:\Laurence\thesis\efizz_chapter\outputs\learning_metrics_per_condition.csv")
A_VS_B_PICKLE = Path(r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\A_vs_B_all_conditions_threat_zone.pkl")

CONDITIONS = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
FEATURE_COLUMNS = [
    "event_speed",
    "event_efficiency",
    "event_is_escape",
    "condition_binary",
]

RANDOM_STATE = 2025
MAX_SPLITS = 5

# -------------------------------------------------------------------------


def p_to_star(p_val: float) -> str:
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


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
    JAL8_21may,
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
    "JAL8_21may",
    "JAL4_3rdSept",
    "JAL4_19thSept",
    "JAL4_28aug",
    "JAL4_11thSept",
]


def load_object(session_path: str, relative_paths: List[List[str]]) -> Dict | None:
    for parts in relative_paths:
        obj_path = os.path.join(session_path, *parts)
        if os.path.exists(obj_path):
            try:
                with open(obj_path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                logger.warning(f"Failed loading {obj_path}: {exc}")
    return None


def build_event_rows(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for session, metrics in results_dict.items():
        if not metrics:
            continue

        def add_events(times, speeds, effs, conditions_arr, event_type: str):
            for cond in CONDITIONS:
                mask = conditions_arr == cond
                if not np.any(mask):
                    continue
                cond_speeds = np.asarray(speeds[mask], dtype=float)
                cond_eff = np.asarray(effs[mask], dtype=float)
                for idx in range(len(cond_speeds)):
                    rows.append(
                        dict(
                            session=session,
                            condition=cond,
                            event_type=event_type,
                            event_speed=float(cond_speeds[idx]),
                            event_efficiency=float(cond_eff[idx]),
                            event_is_escape=int(event_type == "escape"),
                        )
                    )

        add_events(
            np.asarray(metrics.get("timings_homings", []), dtype=float),
            np.asarray(metrics.get("avg_speed_homings", []), dtype=float),
            np.asarray(metrics.get("spatial_efficiency_homings", []), dtype=float),
            np.asarray(metrics.get("conditions_homings", []), dtype=object),
            "homing",
        )
        add_events(
            np.asarray(metrics.get("timings_escapes", []), dtype=float),
            np.asarray(metrics.get("escape_speeds", []), dtype=float),
            np.asarray(metrics.get("spatial_efficiency_escapes", []), dtype=float),
            np.asarray(metrics.get("conditions_escapes", []), dtype=object),
            "escape",
        )
    return pd.DataFrame(rows)


def create_event_dataframe() -> pd.DataFrame:
    data: Dict[str, Dict] = defaultdict(dict)
    for i, experiment in enumerate(experiments_objects):
        session_name = session_NAMES[i]
        loaded_session = get_experiment(experiment)
        session_path = os.path.join(loaded_session.base_path, loaded_session.processed_path)

        homings_obj = load_object(session_path, [["homings", "homings_obj.pkl"]])
        escapes_obj = load_object(session_path, [["escapes", "escapes_obj.pkl"], ["escape", "escapes_obj.pkl"], ["escapes_obj.pkl"]])
        if homings_obj is None or escapes_obj is None:
            logger.warning(f"Missing homing/escape object for {session_name}; skipping.")
            continue

        data[session_name] = dict(
            avg_speed_homings=homings_obj.avg_speed,
            spatial_efficiency_homings=homings_obj.spatial_efficiency,
            conditions_homings=homings_obj.homing_condition,
            timings_homings=[x / 40 for x in homings_obj.onset_frames],
            escape_speeds=escapes_obj.avg_speed,
            spatial_efficiency_escapes=escapes_obj.spatial_efficiency,
            conditions_escapes=escapes_obj.escape_condition,
            timings_escapes=[x / 40 for x in escapes_obj.stim_onset_frames],
        )

    event_df = build_event_rows(data)
    if event_df.empty:
        raise RuntimeError("No event metrics extracted.")

    event_df["homings_mean_speed"] = np.where(
        event_df["event_type"] == "homing",
        event_df["event_speed"],
        np.nan,
    )
    event_df["escapes_mean_speed"] = np.where(
        event_df["event_type"] == "escape",
        event_df["event_speed"],
        np.nan,
    )
    event_df["homings_mean_efficiency"] = np.where(
        event_df["event_type"] == "homing",
        event_df["event_efficiency"],
        np.nan,
    )
    event_df["escapes_mean_efficiency"] = np.where(
        event_df["event_type"] == "escape",
        event_df["event_efficiency"],
        np.nan,
    )
    return event_df


def load_event_metrics() -> pd.DataFrame:
    if REBUILD_EVENT_DATA or not EVENT_CSV.exists():
        logger.info("Rebuilding event-level metrics ...")
        event_df = create_event_dataframe()
        event_df.to_csv(EVENT_CSV, index=False)
    else:
        logger.info("Loading existing event metrics")
        event_df = pd.read_csv(EVENT_CSV)
    return event_df


def aggregate_sessions(final_results: Dict, conditions: List[str]) -> pd.DataFrame:
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
            for flags in clusters.values():
                a_flag = any(flags.get(k, False) for k in ("preflip_tuned", "h_preflipbar_a_tuned", "A_tuned"))
                b_flag = any(flags.get(k, False) for k in ("postflip_tuned", "h_postflipbar_a_tuned", "B_tuned"))
                a_n += int(a_flag)
                b_n += int(b_flag)
                mixed_n += int(flags.get("mixed_tuning", False))
            rows.append(
                dict(
                    session=session,
                    condition=cond,
                    pct_A=a_n / total,
                    pct_B=b_n / total,
                )
            )
    return pd.DataFrame(rows)


def build_labels() -> pd.DataFrame:
    avsb_data = pd.read_pickle(A_VS_B_PICKLE)
    agg_df = aggregate_sessions(avsb_data, CONDITIONS)
    agg_df["delta_pct"] = agg_df["pct_A"] - agg_df["pct_B"]
    agg_df["class_label"] = np.where(
        agg_df["pct_A"] > agg_df["pct_B"],
        1,
        np.where(agg_df["pct_B"] > agg_df["pct_A"], 0, np.nan),
    )
    agg_df = agg_df.dropna(subset=["class_label"]).copy()
    agg_df["class_label"] = agg_df["class_label"].astype(int)
    agg_df["condition_binary"] = (agg_df["condition"] == "barrier_post_flip").astype(int)
    return agg_df[["session", "condition", "delta_pct", "class_label", "condition_binary"]]


def plot_classification(result: Dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["CV accuracy", "Permutation accuracy"],
        [result["cv_mean"], result["permutation_mean"]],
        yerr=[result["cv_std"], result["permutation_std"]],
        color=["#6c5ce7", "#a29bfe"],
        alpha=0.85,
        capsize=6,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Logistic model (AUC={result['auc']:.2f})")
    star = p_to_star(result["permutation_p"])
    y = max(result["cv_mean"] + result["cv_std"], result["permutation_mean"] + result["permutation_std"]) + 0.05
    ax.plot([0, 0, 1, 1], [y - 0.01, y, y, y - 0.01], color="black", linewidth=1.2)
    ax.text(0.5, y + 0.01, star, ha="center", va="bottom")
    fig.tight_layout()
    plt.show()


def plot_regression(result: Dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["CV R²", "Permutation R²"],
        [result["r2_mean"], result["permutation_mean"]],
        yerr=[result["r2_std"], result["permutation_std"]],
        color=["#6c5ce7", "#a29bfe"],
        alpha=0.85,
        capsize=6,
    )
    ax.set_ylabel("R²")
    ax.set_title("OLS regression vs permutation")
    star = p_to_star(result["permutation_p"])
    y = max(result["r2_mean"] + result["r2_std"], result["permutation_mean"] + result["permutation_std"]) + 0.05
    ax.plot([0, 0, 1, 1], [y - 0.01, y, y, y - 0.01], color="black", linewidth=1.2)
    ax.text(0.5, y + 0.01, star, ha="center", va="bottom")
    fig.tight_layout()
    plt.show()


def plot_feature_diagnostics(df: pd.DataFrame, feature_cols: List[str]):
    if df.empty:
        return
    conditions_to_plot = ["barrier_pre_flip", "barrier_post_flip"]
    cond_colors = {"barrier_pre_flip": "#00cec9", "barrier_post_flip": "#fdcb6e"}
    n = len(feature_cols)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for idx, feature in enumerate(feature_cols):
        hist_ax = axes[0, idx]
        for cond in conditions_to_plot:
            vals = (
                df.loc[df["condition"] == cond, feature]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if vals.empty:
                continue
            hist_ax.hist(vals, bins=30, alpha=0.6, color=cond_colors.get(cond, "#95a5a6"), label=cond)
        hist_ax.set_title(f"{feature} by condition")
        hist_ax.set_xlabel(feature)
        if idx == 0:
            hist_ax.set_ylabel("Count")
            hist_ax.legend()

        scatter_ax = axes[1, idx]
        x_vals = df[feature].replace([np.inf, -np.inf], np.nan)
        mask = x_vals.notna() & df["delta_pct"].notna() & df["condition"].isin(conditions_to_plot)
        subset = df.loc[mask]
        if not subset.empty:
            scatter_ax.scatter(
                subset[feature],
                subset["delta_pct"],
                c=subset["condition"].map(cond_colors),
                alpha=0.5,
                edgecolors="none",
            )
        scatter_ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        scatter_ax.set_xlabel(feature)
        if idx == 0:
            scatter_ax.set_ylabel("delta_pct")
        scatter_ax.set_title(f"{feature} vs delta_pct")
    fig.tight_layout()
    plt.show()


def fit_logistic(df: pd.DataFrame) -> Dict:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["class_label"].astype(int).to_numpy()
    groups = df["session"].to_numpy()
    n_splits = max(2, min(MAX_SPLITS, np.unique(groups).size))

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )

    cv = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", groups=groups)
    probs = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, method="predict_proba")
    auc = roc_auc_score(y, probs[:, 1])
    _, perm_scores, perm_p = permutation_test_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_permutations=500,
        n_jobs=1,
        random_state=RANDOM_STATE,
        groups=groups,
    )

    pipeline.fit(X, y)
    return dict(
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        permutation_mean=float(np.mean(perm_scores)),
        permutation_std=float(np.std(perm_scores)),
        permutation_p=float(perm_p),
        auc=float(auc),
    )


def fit_regression(df: pd.DataFrame) -> Dict:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["delta_pct"].to_numpy(dtype=float)
    groups = df["session"].to_numpy()
    n_splits = max(2, min(MAX_SPLITS, np.unique(groups).size))

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )

    cv = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", groups=groups)
    _, perm_scores, perm_p = permutation_test_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_permutations=500,
        n_jobs=1,
        random_state=RANDOM_STATE,
        groups=groups,
    )

    pipeline.fit(X, y)
    return dict(
        r2_mean=float(np.mean(cv_scores)),
        r2_std=float(np.std(cv_scores)),
        permutation_mean=float(np.mean(perm_scores)),
        permutation_std=float(np.std(perm_scores)),
        permutation_p=float(perm_p),
    )


def main():
    event_df = load_event_metrics()
    labels_df = build_labels()
    merged = event_df.merge(labels_df, on=["session", "condition"], how="inner")
    merged = merged.dropna(subset=["class_label", "delta_pct"]).reset_index(drop=True)

    print(f"Total events after merge: {len(merged)}")
    plot_feature_diagnostics(merged, FEATURE_COLUMNS)

    logit_result = fit_logistic(merged)
    print(
        f"Logistic CV accuracy={logit_result['cv_mean']:.3f}±{logit_result['cv_std']:.3f} | "
        f"Permutation accuracy={logit_result['permutation_mean']:.3f}±{logit_result['permutation_std']:.3f} "
        f"(p={logit_result['permutation_p']:.4f}) | AUC={logit_result['auc']:.3f}"
    )
    plot_classification(logit_result)

    reg_result = fit_regression(merged)
    print(
        f"Regression CV R²={reg_result['r2_mean']:.3f}±{reg_result['r2_std']:.3f} | "
        f"Permutation R²={reg_result['permutation_mean']:.3f}±{reg_result['permutation_std']:.3f} "
        f"(p={reg_result['permutation_p']:.4f})"
    )
    plot_regression(reg_result)


if __name__ == "__main__":
    main()
