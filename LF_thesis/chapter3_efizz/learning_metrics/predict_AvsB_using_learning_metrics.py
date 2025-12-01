"""Logistic model predicting whether A tuning dominates B across sessions."""

from __future__ import annotations

from pathlib import Path
import math
import warnings
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------------------------------------------------
CONDITIONS = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
A_KEYS = ["preflip_tuned", "h_preflipbar_a_tuned", "A_tuned"]
B_KEYS = ["postflip_tuned", "h_postflipbar_a_tuned", "B_tuned"]

LEARNING_METRICS_PATH = Path(r"Z:\Laurence\thesis\efizz_chapter\outputs\learning_metrics_per_condition.csv")
A_VS_B_PICKLE = Path(r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\A_vs_B_all_conditions_threat_zone.pkl")

FEATURE_COLUMNS = [
    # "homings_count",
    # "escapes_count",
    "homings_mean_speed",
    "escapes_mean_speed",
    "homings_mean_efficiency",
    "escapes_mean_efficiency",
    # "homings_iei_mean",
    # "escapes_iei_mean",
    # "homing_to_escape_ratio",
]

# ESCAPE COLUMNS ONLY
# FEATURE_COLUMNS = [
#     # "escapes_count",
#     # "escapes_mean_speed",
#     # "escapes_mean_efficiency",
#     # "escapes_median_latency",
#     # "median_escape_times",
#     "mean_escape_times",
# ]

# HOMING COLUMNS ONLY
# FEATURE_COLUMNS = [
#     "homings_count",
#     "homings_mean_speed",
#     "homings_mean_efficiency",
    #"homings_median_latency",
# ]

# -------------- PARAMETERS ----------------

# CONDITIONS = ["barrier_pre_flip"]  # just filter pre flip condition
RANDOM_STATE = 2025  # 2025 for future proofing
BOOTSTRAP_ITER = 200
MAX_SPLITS = 2
C_VALUES = [1.0]  # provide multiple values (e.g., [0.1, 0.5, 1.0]) to sweep
BALANCE_STRATEGY = "downsample"
SCATTER_OUTLIER_ONLY = False 
SCATTER_OUTLIER_Z = 1.44 # 1.645 for 90% confidence
SCATTER_DUAL_PLOTS = True
SCATTER_DUAL_Z = 1.96
FILTER_FEATURE_FOR_MODEL = None  # e.g. "escapes_mean_speed" or None
FILTER_FEATURE_Z = 1
# BALANCE_STRATEGY = "class_weight"



def _flag_value(flags: Dict, candidates: Iterable[str]) -> bool:
    for key in candidates:
        if key in flags and flags.get(key):
            return True
    return False


def p_to_star(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "n.s."
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


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
                a_flag = _flag_value(flags, A_KEYS)
                b_flag = _flag_value(flags, B_KEYS)
                a_n += int(a_flag)
                b_n += int(b_flag)
                mixed_n += int(bool(flags.get("mixed_tuning", False)))

            rows.append(
                dict(
                    session=session,
                    condition=cond,
                    total_cells=total,
                    pct_A=a_n / total,
                    pct_B=b_n / total,
                    pct_mixed=mixed_n / total,
                )
            )
    return pd.DataFrame(rows)


def create_labels(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()
    df["delta_pct"] = df["pct_A"] - df["pct_B"]
    df["class_label"] = np.where(
        df["pct_A"] > df["pct_B"],
        1,
        np.where(df["pct_B"] > df["pct_A"], 0, np.nan),
    )
    df = df.dropna(subset=["class_label"]).copy()
    df["class_label"] = df["class_label"].astype(int)
    df["condition_binary"] = (df["condition"] == "barrier_post_flip").astype(int)
    return df[["session", "condition", "class_label", "condition_binary", "delta_pct"]]


def prepare_dataset() -> Tuple[pd.DataFrame, List[str]]:
    learning_df = pd.read_csv(LEARNING_METRICS_PATH)
    avsb_data = pd.read_pickle(A_VS_B_PICKLE)
    agg_df = aggregate_sessions(avsb_data, CONDITIONS)
    label_df = create_labels(agg_df)

    merged = learning_df.merge(label_df, on=["session", "condition"], how="inner")
    merged = merged[merged["condition"].isin(["barrier_pre_flip", "barrier_post_flip"])].copy()
    merged["condition_binary"] = merged["condition_binary"].astype(int)
    features = [col for col in FEATURE_COLUMNS if col in merged.columns]
    merged["escapes_median_latency"] = merged["escapes_median_latency"].fillna(merged["escapes_median_latency"].median())
    merged["escapes_median_latency"] = merged["escapes_median_latency"] / 40  # scale to seconds

    # if "condition_binary" not in features:
    #     features.append("condition_binary")
    return merged, features


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["class_label"].value_counts()
    min_count = counts.min()
    balanced_parts = []
    for label, count in counts.items():
        subset = df[df["class_label"] == label]
        if count > min_count:
            subset = subset.sample(min_count, random_state=RANDOM_STATE)
        balanced_parts.append(subset)
    balanced_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[INFO] Balanced dataset class counts: {balanced_df['class_label'].value_counts().to_dict()}")
    return balanced_df


def prepare_training_df(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[int, float] | None]:
    strategy = strategy.lower()
    if strategy == "downsample":
        return balance_dataset(df), None

    if strategy == "class_weight":
        shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        y = shuffled["class_label"].to_numpy(dtype=int)
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {int(cls): float(w) for cls, w in zip(classes, weights)}
        print(f"[INFO] Class-weight strategy, weights: {class_weight}")
        return shuffled, class_weight

    raise ValueError(f"Unknown balancing strategy '{strategy}'")


def fit_logit_with_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_bootstrap: int = BOOTSTRAP_ITER,
    alpha: float = 1e-3,
) -> pd.DataFrame:
    X_sm = sm.add_constant(X, has_constant="add")

    def _fit_once(y_vec, X_vec):
        logit = sm.Logit(y_vec, X_vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg_alpha = float(max(alpha, 1e-6))
            return logit.fit_regularized(alpha=reg_alpha, L1_wt=0.0, maxiter=500)

    glm_results = _fit_once(y, X_sm)
    params = np.asarray(glm_results.params, dtype=float)
    coef_df = pd.DataFrame(
        {
            "feature": ["Intercept"] + feature_names,
            "coefficient": params,
        }
    )

    rng = np.random.default_rng(RANDOM_STATE)
    boot_params = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            boot_fit = _fit_once(y[idx], X_sm[idx])
            boot_params.append(np.asarray(boot_fit.params, dtype=float))
        except Exception:
            continue

    if not boot_params:
        ci = np.full((len(feature_names) + 1, 2), np.nan)
        stars = ["n.s."] * (len(feature_names) + 1)
    else:
        boot_params = np.array(boot_params)
        ci = np.percentile(boot_params, [2.5, 97.5], axis=0).T
        p_vals = []
        for j in range(boot_params.shape[1]):
            greater = np.mean(boot_params[:, j] >= 0)
            less = np.mean(boot_params[:, j] <= 0)
            p_val = 2 * min(greater, less)
            if p_val == 0:
                p_val = 1.0 / len(boot_params)
            p_vals.append(min(p_val, 1.0))

        stars = []
        for p in p_vals:
            if p < 0.001:
                stars.append("***")
            elif p < 0.01:
                stars.append("**")
            elif p < 0.05:
                stars.append("*")
            else:
                stars.append("n.s.")

    coef_df["ci_lower"] = ci[:, 0]
    coef_df["ci_upper"] = ci[:, 1]
    coef_df["sig_label"] = stars
    return coef_df


def train_combined_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    balance_strategy: str,
    c_value: float,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    training_df, class_weight = prepare_training_df(df, balance_strategy)
    X = training_df[feature_cols].to_numpy(dtype=float)
    y = training_df["class_label"].to_numpy(dtype=int)
    groups = training_df["session"].to_numpy()
    n_unique_groups = np.unique(groups).size
    n_splits = max(2, min(MAX_SPLITS, n_unique_groups))

    clf_params = dict(max_iter=500, random_state=RANDOM_STATE, penalty="l2", C=c_value)
    if class_weight is not None:
        clf_params["class_weight"] = class_weight

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(**clf_params)),
        ]
    )

    cv = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", groups=groups)
    _, permutation_scores, permutation_p = permutation_test_score(
        pipeline,
        X,
        y,
        cv=cv,
        n_permutations=500,
        scoring="accuracy",
        n_jobs=1,
        random_state=RANDOM_STATE,
        groups=groups,
    )

    pipeline.fit(X, y)
    X_proc = pipeline.named_steps["imputer"].transform(X)
    X_proc = pipeline.named_steps["scaler"].transform(X_proc)
    coef_df = fit_logit_with_bootstrap(X_proc, y, feature_cols, alpha=1.0 / max(c_value, 1e-6))

    cv_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        groups=groups,
    )
    roc_auc = roc_auc_score(y, cv_proba[:, 1])
    fpr, tpr, _ = roc_curve(y, cv_proba[:, 1])
    roc_info = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    result = dict(
        score_mean=float(np.mean(cv_scores)),
        score_std=float(np.std(cv_scores)),
        permutation_mean=float(np.mean(permutation_scores)),
        permutation_std=float(np.std(permutation_scores)),
        permutation_p=float(permutation_p),
        roc_auc=float(roc_auc),
        C=c_value,
        roc_info=roc_info,
    )
    return result, training_df, coef_df, roc_info


def plot_performance_with_coeffs(result: Dict, coef_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 2, 1.2]})

    acc_ax = axes[0]
    bars = ["CV accuracy", "Permutation accuracy"]
    means = [result["score_mean"], result["permutation_mean"]]
    stds = [result["score_std"], result["permutation_std"]]
    x = np.arange(len(bars))
    acc_ax.bar(x, means, yerr=stds, color=["#6c5ce7", "#a29bfe"], alpha=0.85, capsize=6)
    acc_ax.set_xticks(x)
    acc_ax.set_xticklabels(bars, rotation=15)
    acc_ax.set_ylim(0, 1)
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Model vs permutation accuracy")
    star = p_to_star(result["permutation_p"])
    y = max(means[i] + stds[i] for i in range(len(means))) + 0.05
    acc_ax.plot([0, 0, 1, 1], [y - 0.01, y, y, y - 0.01], color="black", linewidth=1.2)
    acc_ax.text(0.5, y + 0.01, star, ha="center", va="bottom")

    coef_ax = axes[1]
    colors = []
    for lower, upper in zip(coef_df["ci_lower"], coef_df["ci_upper"]):
        if np.isnan(lower) or np.isnan(upper):
            colors.append("#95a5a6")
        elif lower > 0:
            colors.append("#2ecc71")
        elif upper < 0:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    for idx, (coef, lower, upper, color) in enumerate(zip(coef_df["coefficient"], coef_df["ci_lower"], coef_df["ci_upper"], colors)):
        left = max(0.0, coef - lower) if np.isfinite(lower) else 0.0
        right = max(0.0, upper - coef) if np.isfinite(upper) else 0.0
        coef_ax.errorbar(
            coef,
            coef_df["feature"].iloc[idx],
            xerr=[[left], [right]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
        )

    coef_ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    xmax = coef_ax.get_xlim()[1]
    for feature, star in zip(coef_df["feature"], coef_df["sig_label"]):
        coef_ax.text(xmax, feature, star, ha="left", va="center")
    coef_ax.set_title("Logit coefficients and bootstrap CIs")
    coef_ax.set_xlabel("Coefficient (log-odds)")

    roc_ax = axes[2]
    roc_info = result.get("roc_info")
    if roc_info:
        roc_ax.plot(roc_info["fpr"], roc_info["tpr"], color="#6c5ce7", lw=2)
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_ax.set_title(f"ROC curve (AUC={roc_info['auc']:.2f})")
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate")
    else:
        roc_ax.text(0.5, 0.5, "ROC unavailable", ha="center")
        roc_ax.set_axis_off()

    fig.tight_layout()
    plt.show()


def plot_feature_collinearity(df: pd.DataFrame, feature_cols: List[str]):
    """Visualize pairwise Pearson correlations to spot colinear features."""
    if not feature_cols:
        return
    corr = df[feature_cols].corr().fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticklabels(feature_cols)
    ax.set_title("Feature correlation heatmap")

    # annotate each cell and highlight highly colinear pairs
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    plt.show()


def plot_feature_vs_delta(
    df: pd.DataFrame,
    feature_cols: List[str],
    outlier_only: bool = SCATTER_OUTLIER_ONLY,
    z_thresh: float = SCATTER_OUTLIER_Z,
) -> pd.DataFrame | None:
    if "delta_pct" not in df.columns or not feature_cols:
        return None
    unique_conditions = sorted(df["condition"].unique())
    colors = {
        "barrier_pre_flip": "#6c5ce7",
        "barrier_post_flip": "#fd79a8",
    }
    outlier_rows = []

    def plot_single(ax, feature: str, only_outliers: bool, thresh: float | None) -> list:
        rows_local = []
        for cond in unique_conditions:
            cond_mask = df["condition"] == cond
            x = df.loc[cond_mask, feature].to_numpy(dtype=float)
            y = df.loc[cond_mask, "delta_pct"].to_numpy(dtype=float)
            sessions = df.loc[cond_mask, "session"].to_numpy(dtype=object)
            finite_mask = np.isfinite(x) & np.isfinite(y)
            x = x[finite_mask]
            y = y[finite_mask]
            sessions = sessions[finite_mask]
            if only_outliers and x.size and thresh is not None:
                x_mean = float(np.mean(x))
                x_std = float(np.std(x))
                y_std = float(np.std(y))
                x_out = np.zeros_like(x, dtype=bool) if x_std == 0 else np.abs((x - x_mean) / x_std) >= thresh
                y_out = np.zeros_like(y, dtype=bool) if y_std == 0 else np.abs(y / y_std) >= thresh
                sel_mask = x_out | y_out
                x = x[sel_mask]
                y = y[sel_mask]
                sessions = sessions[sel_mask]
                for sess, val, del_val in zip(sessions, x, y):
                    rows_local.append(
                        dict(
                            feature=feature,
                            session=str(sess),
                            condition=cond,
                            metric_value=float(val),
                            delta=float(del_val),
                        )
                    )
            ax.scatter(
                x,
                y,
                label=cond.replace("_", " "),
                alpha=0.7,
                color=colors.get(cond, "#636e72"),
            )
            if x.size >= 2 and np.nanmax(x) > np.nanmin(x):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=np.RankWarning)
                        coeffs = np.polyfit(x, y, deg=1)
                    line_x = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                    line_y = np.polyval(coeffs, line_x)
                    ax.plot(line_x, line_y, color=colors.get(cond, "#636e72"), linestyle="-")
                    y_pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    ax.text(
                        0.05,
                        0.9 - 0.1 * unique_conditions.index(cond),
                        f"{cond.replace('_', ' ')} R²={r2:.2f}",
                        transform=ax.transAxes,
                        color=colors.get(cond, "#636e72"),
                        fontsize=9,
                    )
                except np.linalg.LinAlgError:
                    pass
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("pct_A - pct_B")
        title_suffix = "(|z|≥{:.2f})".format(thresh) if only_outliers and thresh is not None else "(all data)"
        ax.set_title(f"{feature} vs tuning delta {title_suffix}")
        return rows_local

    if SCATTER_DUAL_PLOTS:
        n = len(feature_cols)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
        for col, feature in enumerate(feature_cols):
            plot_single(axes[0][col], feature, False, None)
            outlier_rows.extend(plot_single(axes[1][col], feature, True, SCATTER_DUAL_Z))
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        #save fig
        save_dir = r"Z:\Laurence\thesis\efizz_chapter"
        plt.savefig(Path(save_dir) / f"scatter_plots_dual_{SCATTER_DUAL_Z:.2f}.eps")
    else:
        cols = 3
        rows = math.ceil(len(feature_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for idx, feature in enumerate(feature_cols):
            outlier_rows.extend(
                plot_single(
                    axes[idx // cols][idx % cols],
                    feature,
                    outlier_only,
                    z_thresh if outlier_only else None,
                )
            )
        for j in range(len(feature_cols), rows * cols):
            axes[j // cols][j % cols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        plt.show()

    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows)
        # print("\nOutlier sessions used in scatter plots:")
        # print(outlier_df.sort_values(["feature", "condition"]).to_string(index=False))
        return outlier_df
    return None


def filter_dataset_for_model(
    df: pd.DataFrame,
    feature_name: str | None,
    z_thresh: float,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    if not feature_name or feature_name not in df.columns:
        return df.copy(), None

    masks = np.zeros(len(df), dtype=bool)
    rows = []
    for cond in sorted(df["condition"].unique()):
        cond_mask = df["condition"] == cond
        cond_vals = df.loc[cond_mask, feature_name].astype(float)
        finite = cond_vals[np.isfinite(cond_vals)]
        if finite.empty:
            continue
        std = float(finite.std(ddof=0))
        if std == 0 or np.isnan(std):
            continue
        mean = float(finite.mean())
        z_scores = (cond_vals - mean) / std
        selected_idx = z_scores.index[(z_scores.abs() >= z_thresh) & np.isfinite(z_scores)]
        if not selected_idx.empty:
            masks[df.index.isin(selected_idx)] = True
            for idx in selected_idx:
                row = df.loc[idx]
                rows.append(
                    dict(
                        session=row["session"],
                        condition=row["condition"],
                        feature=feature_name,
                        value=row[feature_name],
                        z_score=float(z_scores.loc[idx]),
                    )
                )

    if masks.any():
        filtered = df.loc[masks].reset_index(drop=True)
        return filtered, pd.DataFrame(rows)
    return df.copy(), None


def evaluate_c_grid(merged_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, float]:
    summaries = []
    evaluation_cache = []
    for c_val in C_VALUES:
        result, training_df, coef_df, _ = train_combined_model(merged_df, feature_cols, BALANCE_STRATEGY, c_val)
        coef_df_no_intercept = coef_df[coef_df["feature"] != "Intercept"].reset_index(drop=True)
        sig_mask = coef_df_no_intercept["sig_label"] != "n.s."
        sig_features = ", ".join(coef_df_no_intercept.loc[sig_mask, "feature"]) or "-"
        summaries.append(
            dict(
                C=c_val,
                cv_mean=result["score_mean"],
                cv_std=result["score_std"],
                perm_mean=result["permutation_mean"],
                perm_std=result["permutation_std"],
                perm_p=result["permutation_p"],
                auc=result["roc_auc"],
                num_sig=int(sig_mask.sum()),
                sig_features=sig_features,
            )
        )
        evaluation_cache.append((result, training_df, coef_df_no_intercept))

    summary_df = pd.DataFrame(summaries)
    best_idx = summary_df["cv_mean"].idxmax()
    best_result, best_training, best_coef = evaluation_cache[int(best_idx)]
    best_C = summary_df.loc[best_idx, "C"]
    print("\nC sweep summary:")
    print(summary_df.to_string(index=False))
    return summary_df, best_result, best_training, best_coef, best_C


def main():
    merged_df, feature_cols = prepare_dataset()
    if merged_df.empty or not feature_cols:
        raise RuntimeError("Dataset lacks features or labels.")

    model_df, filter_df = filter_dataset_for_model(merged_df, FILTER_FEATURE_FOR_MODEL, FILTER_FEATURE_Z)
    if FILTER_FEATURE_FOR_MODEL and filter_df is not None:
        print(
            f"\nModel filtering enabled for feature '{FILTER_FEATURE_FOR_MODEL}' "
            f"(z≥{FILTER_FEATURE_Z}); retained {len(model_df)} rows."
        )
        print(filter_df.to_string(index=False))
    elif FILTER_FEATURE_FOR_MODEL:
        print(
            f"\nModel filtering requested for feature '{FILTER_FEATURE_FOR_MODEL}', "
            "but no rows met the z-score threshold. Using full dataset."
        )
        model_df = merged_df

    summary_df, result, training_df, coef_df, best_C = evaluate_c_grid(model_df, feature_cols)
    strategy_desc = "downsampled" if BALANCE_STRATEGY == "downsample" else "class-weighted"
    print(
        f"\nSelected C={best_C} | Training rows ({strategy_desc}): {len(training_df)} | "
        f"CV accuracy={result['score_mean']:.3f}±{result['score_std']:.3f} | "
        f"Permutation accuracy={result['permutation_mean']:.3f}±{result['permutation_std']:.3f} "
        f"(p={result['permutation_p']:.4f})"
    )
    outlier_df = plot_feature_vs_delta(merged_df, feature_cols)
    if outlier_df is not None:
        out_path = Path("LF_thesis/chapter3_efizz/learning_metrics/outputs/scatter_outliers.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        outlier_df.to_csv(out_path, index=False)
    plot_performance_with_coeffs(result, coef_df)
    # plot_feature_collinearity(training_df, feature_cols)


if __name__ == "__main__":
    main()
