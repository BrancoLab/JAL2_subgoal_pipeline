"""OLS model predicting pct_A - pct_B (delta) across conditions."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# -------------------------------------------------------------------------
CONDITIONS = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
A_KEYS = ["preflip_tuned", "h_preflipbar_a_tuned", "A_tuned"]
B_KEYS = ["postflip_tuned", "h_postflipbar_a_tuned", "B_tuned"]

LEARNING_METRICS_PATH = Path(r"Z:\Laurence\thesis\efizz_chapter\outputs\learning_metrics_per_condition.csv")
A_VS_B_PICKLE = Path(r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED\A_vs_B_all_conditions_threat_zone.pkl")

FEATURE_COLUMNS = [
    "homings_count",
    "escapes_count",
    "homings_mean_speed",
    "escapes_mean_speed",
    "homings_mean_efficiency",
    "escapes_mean_efficiency",
    "homings_iei_mean",
    "escapes_iei_mean",
    "homing_to_escape_ratio",
]

RANDOM_STATE = 2025
MAX_SPLITS = 4
BOOTSTRAP_ITER = 400
INCLUDE_CONDITION_BINARY = True

PIPELINE_CONFIGS = [
    dict(name="linear_scaled", use_scaler=True, poly_degree=None, model="linear"),
    dict(name="linear_unscaled", use_scaler=False, poly_degree=None, model="linear"),
    dict(name="ridge_scaled", use_scaler=True, poly_degree=None, model="ridge", alpha=1.0),
    dict(name="ridge_unscaled", use_scaler=False, poly_degree=None, model="ridge", alpha=1.0),
    dict(name="poly2_linear_scaled", use_scaler=True, poly_degree=2, model="linear"),
    dict(name="poly2_ridge_scaled", use_scaler=True, poly_degree=2, model="ridge", alpha=1.0),
]
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
    df["delta_pct"] = df["pct_A"] - df["pct_B"]
    df = df.dropna(subset=["delta_pct"]).copy()
    df["condition_binary"] = (df["condition"] == "barrier_post_flip").astype(int)
    return df[["session", "condition", "delta_pct", "condition_binary"]]


def prepare_dataset() -> Tuple[pd.DataFrame, List[str]]:
    learning_df = pd.read_csv(LEARNING_METRICS_PATH)
    avsb_data = pd.read_pickle(A_VS_B_PICKLE)
    agg_df = aggregate_sessions(avsb_data, CONDITIONS)
    label_df = create_labels(agg_df)

    merged = learning_df.merge(label_df, on=["session", "condition"], how="inner")
    merged = merged[merged["condition"].isin(["barrier_pre_flip", "barrier_post_flip"])].copy()
    features = [col for col in FEATURE_COLUMNS if col in merged.columns]
    if INCLUDE_CONDITION_BINARY and "condition_binary" in merged.columns and "condition_binary" not in features:
        features.append("condition_binary")
    return merged, features


def fit_ols_with_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_bootstrap: int = BOOTSTRAP_ITER,
) -> pd.DataFrame:
    X_sm = sm.add_constant(X, has_constant="add")

    def _fit_once(y_vec, X_vec):
        model = sm.OLS(y_vec, X_vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return model.fit()
            except np.linalg.LinAlgError:
                return model.fit_regularized(alpha=1e-3, L1_wt=0.0)

    ols_results = _fit_once(y, X_sm)
    params = np.asarray(ols_results.params, dtype=float)
    coef_df = pd.DataFrame({"feature": ["Intercept"] + feature_names, "coefficient": params})

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
        stars = [p_to_star(p) for p in p_vals]

    coef_df["ci_lower"] = ci[:, 0]
    coef_df["ci_upper"] = ci[:, 1]
    coef_df["sig_label"] = stars
    return coef_df


def build_pipeline(config: Dict) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.get("poly_degree"):
        steps.append(
            (
                "poly",
                PolynomialFeatures(degree=config["poly_degree"], include_bias=False, interaction_only=False),
            )
        )
    if config.get("use_scaler"):
        steps.append(("scaler", StandardScaler(with_mean=True)))
    model_type = config.get("model", "linear")
    if model_type == "ridge":
        steps.append(("reg", Ridge(alpha=float(config.get("alpha", 1.0)))))
    else:
        steps.append(("reg", LinearRegression()))
    return Pipeline(steps)


def expand_feature_names(feature_cols: List[str], pipeline: Pipeline) -> List[str]:
    names = feature_cols
    if "poly" in pipeline.named_steps:
        poly = pipeline.named_steps["poly"]
        names = poly.get_feature_names_out(names).tolist()
    return names


def train_ols_model(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Dict, pd.DataFrame, Pipeline, pd.DataFrame]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["delta_pct"].to_numpy(dtype=float)
    groups = df["session"].to_numpy()

    n_groups = np.unique(groups).size
    n_splits = max(2, min(MAX_SPLITS, n_groups))

    cfg_summaries = []
    best = None

    for cfg in PIPELINE_CONFIGS:
        pipeline = build_pipeline(cfg)
        cv = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", groups=groups)
        mean_score = float(np.mean(scores))
        cfg_summaries.append(
            dict(name=cfg["name"], mean_r2=mean_score, std_r2=float(np.std(scores)))
        )
        if best is None or mean_score > best["score"]:
            best = dict(cfg=cfg, pipeline=pipeline, score=mean_score)

    print("\nRegression configuration summary (sorted by R²):")
    cfg_table = pd.DataFrame(cfg_summaries).sort_values("mean_r2", ascending=False)
    print(cfg_table.to_string(index=False))

    if best is None:
        raise RuntimeError("No valid pipeline configurations evaluated.")

    best_pipeline = best["pipeline"]
    best_pipeline.fit(X, y)
    cv = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(best_pipeline, X, y, cv=cv, scoring="r2", groups=groups)
    _, permutation_scores, permutation_p = permutation_test_score(
        best_pipeline,
        X,
        y,
        cv=cv,
        n_permutations=500,
        scoring="r2",
        n_jobs=1,
        random_state=RANDOM_STATE,
        groups=groups,
    )

    X_proc = best_pipeline.named_steps["imputer"].transform(X)
    if "poly" in best_pipeline.named_steps:
        X_proc = best_pipeline.named_steps["poly"].transform(X_proc)
    if "scaler" in best_pipeline.named_steps:
        X_proc = best_pipeline.named_steps["scaler"].transform(X_proc)
    expanded_names = expand_feature_names(feature_cols, best_pipeline)
    coef_df = fit_ols_with_bootstrap(X_proc, y, expanded_names)

    result = dict(
        r2_mean=float(np.mean(cv_scores)),
        r2_std=float(np.std(cv_scores)),
        permutation_mean=float(np.mean(permutation_scores)),
        permutation_std=float(np.std(permutation_scores)),
        permutation_p=float(permutation_p),
        best_config=best["cfg"]["name"],
    )
    return result, coef_df, best_pipeline, cfg_table


def plot_results(result: Dict, coef_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

    ax_acc = axes[0]
    bars = ["CV R²", "Permutation R²"]
    means = [result["r2_mean"], result["permutation_mean"]]
    stds = [result["r2_std"], result["permutation_std"]]
    x = np.arange(len(bars))
    ax_acc.bar(x, means, yerr=stds, color=["#6c5ce7", "#a29bfe"], alpha=0.9, capsize=6)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(bars, rotation=15)
    ax_acc.set_ylabel("R²")
    ax_acc.set_ylim(min(-0.5, min(means) - 0.1), 1.0)
    ax_acc.set_title(f"Cross-validated vs. permutation R²\n(best: {result.get('best_config','?')})")
    star = p_to_star(result["permutation_p"])
    y_level = max(means[i] + stds[i] for i in range(len(means))) + 0.05
    ax_acc.plot([0, 0, 1, 1], [y_level - 0.01, y_level, y_level, y_level - 0.01], color="black", linewidth=1.2)
    ax_acc.text(0.5, y_level + 0.01, star, ha="center", va="bottom")

    ax_coef = axes[1]
    for idx, row in coef_df.iterrows():
        feature = row["feature"]
        coef = row["coefficient"]
        lower = row["ci_lower"]
        upper = row["ci_upper"]
        color = "#95a5a6"
        if np.isfinite(lower) and np.isfinite(upper):
            if lower > 0:
                color = "#2ecc71"
            elif upper < 0:
                color = "#e74c3c"
        ax_coef.errorbar(
            coef,
            feature,
            xerr=[[coef - lower if np.isfinite(lower) else 0], [upper - coef if np.isfinite(upper) else 0]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            elinewidth=2,
        )
    ax_coef.axvline(0, color="black", linestyle="--", linewidth=0.8)
    xmax = ax_coef.get_xlim()[1]
    for feature, star_label in zip(coef_df["feature"], coef_df["sig_label"]):
        ax_coef.text(xmax, feature, star_label, ha="left", va="center")
    ax_coef.set_title("OLS coefficients with bootstrap CI")
    ax_coef.set_xlabel("Coefficient (delta units)")

    fig.tight_layout()
    plt.show()


def main():
    df, feature_cols = prepare_dataset()
    if df.empty or not feature_cols:
        raise RuntimeError("Dataset lacks required features or labels.")

    result, coef_df, _, _ = train_ols_model(df, feature_cols)
    print(
        f"Samples: {len(df)} | Best config={result['best_config']} | "
        f"CV R²={result['r2_mean']:.3f}±{result['r2_std']:.3f} | "
        f"Permutation R²={result['permutation_mean']:.3f}±{result['permutation_std']:.3f} "
        f"(p={result['permutation_p']:.4f})"
    )
    plot_results(result, coef_df)


if __name__ == "__main__":
    main()
