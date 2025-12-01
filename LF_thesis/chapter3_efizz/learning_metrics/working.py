"""Logistic model predicting whether A tuning dominates B across sessions."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
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
    "escapes_count",
    # "homings_mean_speed",
    "escapes_mean_speed",
    # "homings_mean_efficiency",
    "escapes_mean_efficiency",
    # "homings_speed_cv",
    # "escapes_speed_cv",
    # "homings_efficiency_std",
    # "escapes_efficiency_std",
    # "homings_rate_per_10min",
    # "escapes_rate_per_10min",
    #"homings_median_latency", (Super correlated with escapes_median_latency and im not sure what even is homing latency so dont include)
    "escapes_median_latency",
    # "homings_iei_mean",
    # "escapes_iei_mean",
    # "homing_to_escape_ratio",
]

# ESCAPE COLUMNS ONLY
FEATURE_COLUMNS = [
    "escapes_count",
    # "escapes_mean_speed",
    "escapes_mean_efficiency",
    "escapes_median_latency",
]

# HOMING COLUMNS ONLY
# FEATURE_COLUMNS = [
#     "homings_count",
#     "homings_mean_speed",
#     "homings_mean_efficiency",
#     "homings_median_latency",
# ]

CONDITIONS = ["barrier_pre_flip"] # just filter pre flip condition
RANDOM_STATE = 2025 # 2025 for future proofing
BOOTSTRAP_ITER = 200
MAX_SPLITS = 2
C = 1
BALANCE_STRATEGY = "downsample"  # or "class_weight"
# BALANCE_STRATEGY = "class_weight"
# -------------------------------------------------------------------------


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
    df["class_label"] = np.where(
        df["pct_A"] > df["pct_B"],
        1,
        np.where(df["pct_B"] > df["pct_A"], 0, np.nan),
    )
    df = df.dropna(subset=["class_label"]).copy()
    df["class_label"] = df["class_label"].astype(int)
    df["condition_binary"] = (df["condition"] == "barrier_post_flip").astype(int)
    return df[["session", "condition", "class_label", "condition_binary"]]


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
    merged["escapes_median_latency"] = merged["escapes_median_latency"] / 40 # scale to seconds

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
) -> pd.DataFrame:
    X_sm = sm.add_constant(X, has_constant="add")

    def _fit_once(y_vec, X_vec):
        logit = sm.Logit(y_vec, X_vec)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = logit.fit(disp=0, maxiter=500)
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_glm = sm.GLM(y_vec, X_vec, family=sm.families.Binomial())
                res = ridge_glm.fit_regularized(alpha=1e-3, L1_wt=0.0, maxiter=500)
        return res

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
    balance_strategy: str = BALANCE_STRATEGY,
) -> Tuple[Dict, pd.DataFrame]:
    training_df, class_weight = prepare_training_df(df, balance_strategy)
    X = training_df[feature_cols].to_numpy(dtype=float)
    y = training_df["class_label"].to_numpy(dtype=int)
    min_class = np.bincount(y).min()
    n_splits = min(MAX_SPLITS, min_class)

    clf_params = dict(max_iter=500, random_state=RANDOM_STATE, penalty="l2", C=C)
    if class_weight is not None:
        clf_params["class_weight"] = class_weight

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(**clf_params)),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    _, permutation_scores, permutation_p = permutation_test_score(
        pipeline,
        X,
        y,
        cv=cv,
        n_permutations=500,
        scoring="accuracy",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    pipeline.fit(X, y)
    X_proc = pipeline.named_steps["imputer"].transform(X)
    X_proc = pipeline.named_steps["scaler"].transform(X_proc)
    coef_df = fit_logit_with_bootstrap(X_proc, y, feature_cols)

    result = dict(
        score_mean=float(np.mean(cv_scores)),
        score_std=float(np.std(cv_scores)),
        permutation_mean=float(np.mean(permutation_scores)),
        permutation_std=float(np.std(permutation_scores)),
        permutation_p=float(permutation_p),
    )
    return result, training_df, coef_df


def plot_performance_with_coeffs(result: Dict, coef_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

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

def main():
    merged_df, feature_cols = prepare_dataset()
    if merged_df.empty or not feature_cols:
        raise RuntimeError("Dataset lacks features or labels.")

    result, training_df, coef_df = train_combined_model(merged_df, feature_cols, BALANCE_STRATEGY)
    coef_df = coef_df[coef_df["feature"] != "Intercept"].reset_index(drop=True)
    strategy_desc = "downsampled" if BALANCE_STRATEGY == "downsample" else "class-weighted"
    print(
        f"Training rows ({strategy_desc}): {len(training_df)} | "
        f"CV accuracy={result['score_mean']:.3f}±{result['score_std']:.3f} | "
        f"Permutation accuracy={result['permutation_mean']:.3f}±{result['permutation_std']:.3f} "
        f"(p={result['permutation_p']:.4f})"
    )
    plot_performance_with_coeffs(result, coef_df)
    plot_feature_collinearity(training_df, feature_cols)



if __name__ == "__main__":
    main()
