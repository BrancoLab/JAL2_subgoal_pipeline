"""Predict which neurons adopt the post-flip tuning.

Workflow
--------
1. Load the per-cluster Rayleigh summary table for threat trials.
2. For each neuron, select the pre/post condition with the highest Rayleigh
   magnitude and compute interpretable pre-condition statistics.
3. Fit a regularized logistic regression (balanced class weights) to predict
   whether the post condition is `h_postflipbar_a` with strong Rayleigh support.
4. Bootstrap coefficients to obtain confidence intervals (no filtering / NaNs).
5. Quantify accuracy via bootstrap and permutation tests and plot the null
   distribution alongside the observed value (with ROC-AUC annotation).
"""

import os
import sys
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
    warnings.warn("scipy not available: paired tests will be skipped.")

# --- CONFIG ---
dir = r"Z:\Jasmine_Laurence\rayleigh_analysis"
csv_path = os.path.join(dir, "threat_dict_max_rayleigh_flat.csv")
n_perm = 10000
alpha = 0.05
MIN_CLASS_COUNT = 20  # minimum samples per indicator to keep it
A_pre = "h_preflipbar_a"
pre_cond, post_cond = "barrier_pre_flip", "barrier_post_flip"
BALANCE_CLASSES = True  # Toggle to downsample classes to 50/50
BALANCE_SEED = 0

# --- LOAD ---
if not os.path.exists(csv_path):
    print(f"CSV not found: {csv_path}")
    sys.exit(1)
df_long = pd.read_csv(csv_path)

# ...existing code...
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Fallbacks to reuse your helpers if not already defined
if 'find_col' not in globals():
    def find_col(df, candidates):
        """Return the first matching column name from a list of candidates."""
        m = {str(c).lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in m:
                return m[c.lower()]
        raise KeyError(f"Missing any of: {candidates}")

if 'normalize_angle_label' not in globals():
    def normalize_angle_label(s: str) -> str:
        """Map multiple label variants onto canonical names."""
        s = str(s).lower()
        if "postflip" in s or "post_flip" in s: return "h_postflipbar_a"
        if "preflip"  in s or "pre_flip"  in s: return "h_preflipbar_a"
        if "hdir" in s: return "hdir"
        if "hsa"  in s: return "hsa"
        return s

def _find_rayleigh_col(_df):
    """Find the column containing Rayleigh magnitudes."""
    for c in _df.columns:
        if "rayleigh" in str(c).lower():
            return c
    raise KeyError("Rayleigh column not found in df.")

if 'to_float_list' not in globals():
    import ast, re
    def to_float_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
        elif isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in ("nan","none","null"):
                return []
            try:
                arr = np.asarray(ast.literal_eval(s), dtype=float)
            except Exception:
                parts = [p for p in re.split(r"[,;\s]+", s.strip("[]")) if p]
                try:
                    arr = np.asarray([float(p) for p in parts], dtype=float)
                except Exception:
                    return []
        else:
            return []
        if arr.ndim > 1: arr = arr.ravel()
        arr = arr[np.isfinite(arr)]
        return arr.tolist()

def _fr_stats_from_list(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=np.nan, cv=np.nan, frac_zero=np.nan, max=np.nan)
    mean = float(np.nanmean(a))
    std  = float(np.nanstd(a, ddof=1)) if a.size > 1 else 0.0
    cv   = float(std/mean) if mean != 0 else np.nan
    frac_zero = float(np.mean(a <= 1e-12))
    return dict(mean=mean, cv=cv, frac_zero=frac_zero, max=float(np.nanmax(a)))

def _cosine_similarity(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = min(a.size, b.size)
    if L == 0: return np.nan
    a = a[:L]; b = b[:L]
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float(np.dot(a, b) / (na * nb))


def make_balanced_weights(y, groups):
    """Return per-sample weights that balance class labels and sessions."""
    y = pd.Series(y).astype(int)
    N = len(y)
    n1 = y.sum(); n0 = N - n1
    w_class = np.where(y == 1, (N/(2*n1)) if n1 > 0 else 1.0,
                              (N/(2*n0)) if n0 > 0 else 1.0)
    g = pd.Series(groups)
    g_counts = g.value_counts()
    w_group = g.map(N / (len(g_counts) * g_counts))
    return np.asarray(w_class) * np.asarray(w_group)


def downsample_balanced_indices(y, seed=0):
    """Return indices that downsample classes to equal counts."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return np.arange(len(y))
    min_count = counts.min()
    rng = np.random.default_rng(seed)
    idxs = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size <= min_count:
            idxs.append(cls_idx)
        else:
            idxs.append(rng.choice(cls_idx, size=min_count, replace=False))
    keep = np.sort(np.concatenate(idxs))
    return keep


def standardize_df(X: pd.DataFrame):
    """Return z-scored features (mean 0, std 1)."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.values.astype(float))
    Z = pd.DataFrame(Z, columns=X.columns, index=X.index)
    return Z, scaler


def prune_constant_features(df: pd.DataFrame, columns):
    """Drop features that become constant after filtering to avoid singular fits."""
    kept = []
    for col in columns:
        unique = df[col].nunique(dropna=True)
        if unique <= 1:
            print(f"[INFO] Removing constant feature: {col}")
        else:
            kept.append(col)
    return kept


def build_classifier(C=1.0):
    """Return a balanced logistic-regression classifier with L2 penalty."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        penalty="l2",
        C=C,
    )


def train_classifier(X, y, C=1.0):
    clf = build_classifier(C)
    clf.fit(X, y)
    return clf


def bootstrap_coefficients(X, y, n_boot=500, seed=0, C=1.0):
    """Bootstrap coefficients by resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    coefs = np.zeros((n_boot, X.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        clf = build_classifier(C)
        clf.fit(X[idx], y[idx])
        coefs[i] = clf.coef_[0]
    return coefs

def _build_top_pre_post(df, pre_cond="barrier_pre_flip", post_cond="barrier_post_flip"):
    """Return the top (max Rayleigh) threat entries for pre and post conditions."""
    session_col     = find_col(df, ["session","session_name","sesh"])
    cluster_col     = find_col(df, ["cluster_id","cluster","cell_id","unit","neuron"])
    condition_col   = find_col(df, ["condition"])
    compartment_col = find_col(df, ["compartment","zone"])
    angle_col       = find_col(df, ["angle","angle_name","angle_key"])
    rayleigh_col    = _find_rayleigh_col(df)
    fr_col          = "firing_rate_hz"

    # THREAT + pre/post
    d = df[(df[compartment_col].astype(str).str.lower()=="threat") &
           (df[condition_col].isin([pre_cond, post_cond]))].copy()
    if d.empty:
        raise SystemExit("No THREAT rows for target conditions.")

    # Choose top (max Rayleigh) row per (session, cluster, condition)
    idx = d.groupby([session_col, cluster_col, condition_col])[rayleigh_col].idxmax()
    top = d.loc[idx].copy()

    pre  = top[top[condition_col].eq(pre_cond)].copy()
    post = top[top[condition_col].eq(post_cond)].copy()
    m = pre.merge(post, on=[session_col, cluster_col], suffixes=("_pre","_post"))
    return dict(df=d, top_pre=pre, top_post=post, merged=m,
                session=session_col, cluster=cluster_col, cond=condition_col,
                comp=compartment_col, angle=angle_col, ray=rayleigh_col, fr=fr_col)

def build_model_table(df, ray_cut=0.25):
    """Construct the modeling dataframe and feature list."""
    meta = _build_top_pre_post(df)
    m    = meta["merged"]
    ang  = meta["angle"]; ray = meta["ray"]; fr = meta["fr"]
    ses  = meta["session"]

    # Parse FR vectors and compute stats
    pre_vecs  = m[f"{fr}_pre"].apply(to_float_list)
    post_vecs = m[f"{fr}_post"].apply(to_float_list)

    pre_stats  = pre_vecs.apply(_fr_stats_from_list).apply(pd.Series)
    post_stats = post_vecs.apply(_fr_stats_from_list).apply(pd.Series)

    # Cosine similarity (pattern stability)
    cos = [ _cosine_similarity(np.asarray(a,float), np.asarray(b,float))
            for a,b in zip(pre_vecs, post_vecs) ]

    out = pd.DataFrame({
        "session": m[ses].values,
        "pre_angle":  m[f"{ang}_pre"].map(normalize_angle_label).values,
        "post_angle": m[f"{ang}_post"].map(normalize_angle_label).values,
        "rayleigh_pre":  m[f"{ray}_pre"].astype(float).values,
        "rayleigh_post": m[f"{ray}_post"].astype(float).values,
        "fr_mean_pre":  pre_stats["mean"].values,
        "fr_mean_post": post_stats["mean"].values,
        "fr_cv_pre":    pre_stats["cv"].values,
        "fr_cv_post":   post_stats["cv"].values,
        "fr_fraczero_pre":  pre_stats["frac_zero"].values,
        "fr_fraczero_post": post_stats["frac_zero"].values,
        "fr_max_pre":   pre_stats["max"].values,
        "fr_max_post":  post_stats["max"].values,
        "cosine_similarity": np.asarray(cos, float)
    })

    # Targets
    out["Y_post_is_postflip"] = (out["post_angle"].eq("h_postflipbar_a")).astype(int)
    out["Y_post_is_postflip_strict"] = ((out["post_angle"].eq("h_postflipbar_a")) &
                                        (out["rayleigh_post"] > float(ray_cut))).astype(int)
    # "New" cells coming into postflip from other pre angles
    out["Y_new_to_postflip"] = ((~out["pre_angle"].eq("h_postflipbar_a")) &
                                (out["post_angle"].eq("h_postflipbar_a")) &
                                (out["rayleigh_post"] > float(ray_cut))).astype(int)

    # One-hot indicators for specific pre angles of interest
    indicator_map = {
        "pre_h_postflipbar_a": "h_postflipbar_a",
        "pre_h_preflipbar_a": "h_preflipbar_a",
        "pre_hsa": "hsa",
        "pre_hdir": "hdir",
    }
    retained_indicators = []
    for col, label in indicator_map.items():
        vals = out["pre_angle"].eq(label).astype(int)
        positives = int(vals.sum())
        if positives < MIN_CLASS_COUNT or positives == len(vals):
            print(f"[INFO] Dropping indicator {col} (count={positives})")
            continue
        out[col] = vals
        retained_indicators.append(col)

    # Feature matrix (include most of what we built; exclude direct leakage post-only angle label)
    feature_cols = [
        "rayleigh_pre",
        "fr_mean_pre",
        "fr_fraczero_pre",
    ] + retained_indicators

    # Keep finite rows
    mask = np.isfinite(out[feature_cols].to_numpy(dtype=float)).all(axis=1)
    model_df = out.loc[mask].reset_index(drop=True)
    return model_df, feature_cols

model_df, feats = build_model_table(df_long if 'df_long' in globals() else df)

print("Rows in model table:", len(model_df))
print("Class balance:")
for col in ["Y_post_is_postflip","Y_post_is_postflip_strict","Y_new_to_postflip"]:
    if col in model_df:
        p = model_df[col].mean()
        print(f"  {col}: mean={p:.3f} (n1={int(model_df[col].sum())}, n0={int((1-model_df[col]).sum())})")

feats = prune_constant_features(model_df, feats)

# === Reliable, bias-aware modeling + coefficient plots ===

def fit_glm_weighted(Xz, y, groups, label=""):
    """Fit a weighted logistic GLM with cluster-robust covariance."""
    Xc = sm.add_constant(Xz, has_constant="add")
    w  = make_balanced_weights(y, groups)
    model = sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=w)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    print(f"\n=== {label} ===")
    print(f"n={len(y)}, pos={int(np.sum(y))}, neg={int(len(y)-np.sum(y))}")
    print(res.summary())
    return res

def group_cv_auc_pr(X, y, groups, C=1.0):
    """Estimate out-of-sample AUC/PR using GroupKFold splits."""
    pipe = make_pipeline(StandardScaler(), build_classifier(C))
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    auc = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="roc_auc")
    pr  = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="average_precision")
    return auc.mean(), auc.std(), pr.mean(), pr.std()

def plot_coefficients(ax, coef_mean, ci_low, ci_high, feature_names, title, boot_coefs=None):
    """Plot coefficients with bootstrap confidence intervals."""
    order = np.argsort(np.abs(coef_mean))[::-1]
    coef_mean = coef_mean[order]
    ci_low = ci_low[order]
    ci_high = ci_high[order]
    feature_names = np.array(feature_names)[order]
    boot = None if boot_coefs is None else boot_coefs[:, order]

    y = np.arange(len(coef_mean))[::-1]
    colors = np.where(coef_mean > 0, "#d62728", "#1f77b4")
    ax.hlines(y, ci_low, ci_high, color="#555", lw=2)
    ax.scatter(coef_mean, y, c=colors, s=60, zorder=3,
               edgecolor="white", linewidth=0.7)
    if boot is not None:
        pvals = []
        for idx in range(len(feature_names)):
            sign = np.sign(coef_mean[idx])
            if sign == 0:
                pvals.append(1.0)
                continue
            more_extreme = np.mean(sign * boot[:, idx] <= 0)
            pvals.append(min(more_extreme, 1 - more_extreme) * 2)
        signif = np.select(
            [np.array(pvals) < 0.001, np.array(pvals) < 0.01, np.array(pvals) < 0.05],
            ["***", "**", "*"],
            default="",
        )
        for yi, (coef, sig) in enumerate(zip(coef_mean, signif)):
            if sig:
                ax.text(coef, y[yi], f" {sig}", va="center", ha="left", fontsize=10)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Standardized log-odds coefficient (bootstrap 95% CI)")
    ax.set_title(title)
    return feature_names

def accuracy_permutation_plot(ax, y, preds, n_boot=1000, n_perm=1000, seed=0, auc_score=None):
    rng = np.random.default_rng(seed)
    hard = (preds > 0.5).astype(int)
    n = len(y)
    acc = accuracy_score(y, hard)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = accuracy_score(y[idx], hard[idx])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = accuracy_score(rng.permutation(y), hard)
    p_value = (np.sum(perm >= acc) + 1) / (n_perm + 1)
    star = (
        "***" if p_value < 0.001 else
        "**" if p_value < 0.01 else
        "*" if p_value < 0.05 else
        "n.s."
    )

    ax.hist(perm, bins=20, color="#b0b0b0", edgecolor="white", alpha=0.6, label="Permutation null")
    ax.hist(boot, bins=20, color="#7fb3d5", edgecolor="white", alpha=0.5, label="Bootstrap resamples")
    ax.axvline(acc, color="#c0392b", lw=2, label="Observed accuracy")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title("Accuracy significance (permutation + bootstrap)")
    auc_text = f"{auc_score:.3f}" if auc_score is not None else "n/a"
    text = (
        f"Observed accuracy: {acc:.3f}\n"
        f"95% bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"Permutation p-value: {p_value:.3f} ({star})\n"
        f"ROC AUC: {auc_text}\n"
        f"n={n}, boot={n_boot}, perm={n_perm}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="upper right", frameon=False)

def plot_conversion_vs_fr_mean_bins(df: pd.DataFrame, y_col: str, n_bins: int = 8) -> None:
    """Bin fr_mean_pre and show the conversion fraction in each bin."""
    data = df[["fr_mean_pre", y_col]].dropna().copy()
    if data.empty:
        print("[WARN] No finite fr_mean_pre values; skipping bin plot.")
        return
    data["fr_bin"] = pd.qcut(data["fr_mean_pre"], q=n_bins, duplicates="drop")
    stats = (
        data.groupby("fr_bin", observed=True)
        .agg(conv=(y_col, "mean"), count=(y_col, "size"), mid=("fr_mean_pre", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(stats["mid"], stats["conv"], marker="o", color="#2c3e50")
    ax.set_xlabel("fr_mean_pre (bin centre)")
    ax.set_ylabel(f"P({y_col} = 1)")
    ax.set_title("Conversion fraction vs fr_mean_pre quantile")
    for _, row in stats.iterrows():
        ax.text(row["mid"], row["conv"] + 0.02, f"n={int(row['count'])}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_fr_mean_distribution_by_class(df: pd.DataFrame, y_col: str) -> None:
    """Compare fr_mean_pre distributions between classes."""
    data = df[["fr_mean_pre", y_col]].dropna().copy()
    if data.empty:
        print("[WARN] No data for fr_mean_pre distribution plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    sns.histplot(
        data=data,
        x="fr_mean_pre",
        hue=y_col,
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title("fr_mean_pre distribution by class")
    ax.set_xlabel("fr_mean_pre")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.show()

def plot_fr_mean_logistic(df: pd.DataFrame, y_col: str) -> None:
    """Show raw relationship between fr_mean_pre and the binary target."""
    data = df[["fr_mean_pre", y_col]].dropna().copy()
    if data.empty:
        print("[WARN] No data for fr_mean_pre logistic plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    sns.regplot(
        x="fr_mean_pre",
        y=y_col,
        data=data,
        logistic=True,
        scatter_kws={"s": 20, "alpha": 0.3},
        line_kws={"color": "#c0392b"},
        ax=ax,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("fr_mean_pre")
    ax.set_ylabel(f"P({y_col} = 1)")
    ax.set_title("Logistic trend of fr_mean_pre vs conversion")
    plt.tight_layout()
    plt.show()

def plot_ecdf_pair(
    df: pd.DataFrame,
    features: list[str],
    y_col: str,
    legend_title: Optional[str] = None,
) -> None:
    """Plot ECDFs for multiple features in a single figure with per-axis legends."""
    valid_feats = [feat for feat in features if feat in df.columns]
    if not valid_feats:
        print("[WARN] No valid features for ECDF plot.")
        return
    fig, axes = plt.subplots(
        1,
        len(valid_feats),
        figsize=(6 * len(valid_feats), 4),
        dpi=140,
        sharey=True,
    )
    if len(valid_feats) == 1:
        axes = [axes]
    legend_title = legend_title or y_col
    for ax, feature in zip(axes, valid_feats):
        data = df[[feature, y_col]].dropna().copy()
        if data.empty:
            ax.set_visible(False)
            continue
        sns.ecdfplot(data=data, x=feature, hue=y_col, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of {feature} by class")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            pretty_labels = [f"{y_col} = {lab}" for lab in labels]
            ax.legend(
                handles,
                pretty_labels,
                title=legend_title,
                loc="lower right",
                frameon=False,
            )
    plt.tight_layout()
    plt.show()

# Prepare features and groups
X = model_df[feats].astype(float)
groups = model_df["session"].values
Xz, _scaler = standardize_df(X)

targets = {
    "Y_post_is_postflip_strict": "Post angle = postflip AND Rayleigh_post > cut"
}

X_array = Xz.values
X_raw_array = X.values

for y_col, y_label in targets.items():
    if y_col not in model_df:
        print(f"Missing target {y_col}; skipping.")
        continue
    y_full = model_df[y_col].astype(int).values
    if np.unique(y_full).size < 2:
        print(f"[WARN] Target {y_col} has <2 classes; skipping.")
        continue

    idx = np.arange(len(y_full))
    if BALANCE_CLASSES:
        idx = downsample_balanced_indices(y_full, seed=BALANCE_SEED)
        counts = dict(zip(*np.unique(y_full[idx], return_counts=True)))
        print(f"[INFO] Downsampled {y_col} to balanced classes: {counts}")

    y = y_full[idx]
    groups_used = groups[idx]
    X_bal = X_array[idx]
    X_raw_bal = X_raw_array[idx]
    df_bal = model_df.iloc[idx].reset_index(drop=True)

    clf = train_classifier(X_bal, y)
    preds = clf.predict_proba(X_bal)[:, 1]
    coeff_boot = bootstrap_coefficients(X_bal, y, n_boot=500)
    ci_low, ci_high = np.percentile(coeff_boot, [2.5, 97.5], axis=0)

    # plot_conversion_vs_fr_mean_bins(df_bal, y_col, n_bins=8)
    # plot_fr_mean_distribution_by_class(df_bal, y_col)
    # plot_fr_mean_logistic(df_bal, y_col)
    auc_m, auc_s, pr_m, pr_s = group_cv_auc_pr(X_raw_bal, y, groups_used)
    print(f"Group-CV AUC: {auc_m:.3f} +/- {auc_s:.3f} | PR-AUC: {pr_m:.3f} +/- {pr_s:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    plot_coefficients(
        axes[0],
        clf.coef_[0],
        ci_low,
        ci_high,
        feats,
        f"Feature effects ({y_label})",
        boot_coefs=coeff_boot,
    )
    accuracy_permutation_plot(
        axes[1],
        y,
        preds,
        n_boot=1000,
        n_perm=1000,
        auc_score=roc_auc_score(y, preds),
    )
    plt.tight_layout()
    plt.show()
    plot_ecdf_pair(df_bal, ["fr_mean_pre", "rayleigh_pre"], y_col, y_label)




