# --- CONFIG ---
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir = r"Z:\Jasmine_Laurence\rayleigh_analysis"
csv_path = os.path.join(dir, "threat_dict_max_rayleigh_flat.csv")
target_key = "h_preflipbar_a"     # test whether its Δ < 0 relative to random label shuffles
n_perm = 10000
alpha = 0.05

# --- LOAD ---
df = pd.read_csv(csv_path)

# --- HELPERS ---
def find_col(df, candidates):
    m = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    raise KeyError(f"Missing any of: {candidates}")

def normalize_angle_label(s: str) -> str:
    s = str(s).lower()
    if "postflip" in s or "post_flip" in s: return "h_postflipbar_a"
    if "preflip"  in s or "pre_flip"  in s: return "h_preflipbar_a"
    if "hdir" in s: return "hdir"
    if "hsa"  in s: return "hsa"
    return s

def perm_test_against_zero(x, n_perm=10000, seed=7, tail="less"):
    """Permutation test for mean(x) < 0 (one-sided)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0: 
        return np.nan
    rng = np.random.default_rng(seed)
    obs = x.mean()
    # create null by randomly flipping signs
    sims = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(x))
        sims[i] = (x * signs).mean()
    p = (np.sum(sims <= obs) + 1) / (n_perm + 1) if tail == "less" else (np.sum(np.abs(sims) >= abs(obs)) + 1)/(n_perm+1)
    return p, sims, obs

# --- COLUMNS ---
session_col     = find_col(df, ["session","session_name","sesh"])
cluster_col     = find_col(df, ["cluster_id","cluster","cell_id","unit","neuron"])
condition_col   = find_col(df, ["condition"])
compartment_col = find_col(df, ["compartment","zone"])
angle_col       = find_col(df, ["angle","angle_name","angle_key"])
rayleigh_col    = next(c for c in df.columns if "rayleigh" in str(c).lower())

# --- FILTER: threat + pre/post only ---
pre_cond, post_cond = "barrier_pre_flip", "barrier_post_flip"
df_threat = df[
    (df[compartment_col].astype(str).str.lower() == "threat") &
    (df[condition_col].isin([pre_cond, post_cond]))
].copy()

# --- Top (max Rayleigh) per session×cell×condition ---
idx = df_threat.groupby([session_col, cluster_col, condition_col])[rayleigh_col].idxmax()
top = df_threat.loc[idx].copy()
top_pre  = top[top[condition_col].eq(pre_cond)].copy()
top_post = top[top[condition_col].eq(post_cond)].copy()

# --- Keep cells pre-top == A_pre ---
preA_cells = top_pre.loc[top_pre[angle_col].map(normalize_angle_label).eq(target_key),
                         [session_col, cluster_col]]
merged = (top_pre.merge(top_post, on=[session_col, cluster_col], suffixes=("_pre","_post"))
                .merge(preA_cells, on=[session_col, cluster_col], how="inner"))
if merged.empty:
    raise SystemExit("No matched pre/post rows for pre-top = A_pre cells in THREAT.")

# --- Compute Δ ---
r_pre  = merged[f"{rayleigh_col}_pre"].astype(float).to_numpy()
r_post = merged[f"{rayleigh_col}_post"].astype(float).to_numpy()
delta  = r_post - r_pre

# --- Permutation test (sign-flip null) ---
p_val, null_means, obs_mean = perm_test_against_zero(delta, n_perm=n_perm, seed=11, tail="less")
ci_lo, ci_hi = np.percentile(null_means, [2.5, 97.5])
print(f"\nPermutation test: mean(Δ) < 0 ?")
print(f"Observed mean Δ = {obs_mean:.4f}")
print(f"95% null CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"p_perm(one-sided, less) = {p_val:.4g}")

plt.figure(figsize=(8,4.8))

# --- Normalised probability version ---
plt.figure(figsize=(8,4.8))

# 1️⃣ Empirical Δ distribution (actual per-cell changes)
sns.kdeplot(
    delta, fill=True, color="#3b6ba5", lw=1.8, alpha=0.35,
    label="Per-cell Δ (post − pre)", common_norm=False, bw_adjust=0.7
)

# 2️⃣ Null distribution of permuted mean values
sns.kdeplot(
    null_means, fill=True, color="#cccccc", lw=1.2, alpha=0.5,
    label="Null mean (sign shuffles)", common_norm=False, bw_adjust=0.7
)

# 3️⃣ Observed mean and 95 % CI
plt.axvline(obs_mean, color="red", lw=2.2, label=f"Observed mean Δ = {obs_mean:.3f}")
plt.axvline(ci_lo, color="black", ls="--", lw=1)
plt.axvline(ci_hi, color="black", ls="--", lw=1)

# 4️⃣ Zoom around main spread
xmin, xmax = np.percentile(delta, [0.1, 99.9])
plt.xlim(xmin, xmax)

plt.title(f"Normalised Δ Rayleigh distribution (pre-top = h_preflipbar_a)\nPermutation p = {p_val:.4g}, n = {len(delta)}")
plt.xlabel("Δ Rayleigh (post − pre)")
plt.ylabel("Probability Density")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
