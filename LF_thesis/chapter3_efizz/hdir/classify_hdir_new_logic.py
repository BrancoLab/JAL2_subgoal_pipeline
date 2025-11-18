# Head-direction cells per mouse across sessions (robust mouse inference + skip summary)
"""
Classify head-direction (HD) cells across sessions by comparing each cell's Rayleigh magnitude and preferred angle
between shelter/threat compartments and aggregating counts per mouse.

Procedure summary:
* Load the precomputed Rayleigh data (magnitudes, preferred angles, significance flags) for each session.
* For every cluster with finite values, compute permutation distributions by shuffling shelter and threat data across cells.
* Use the geometric mean Rayleigh magnitude and circular angle difference as the test statistics.
* Derive per-cell p-values from the shuffle distributions:
    - `rayleigh_p`: probability that shuffled geometric means exceed the actual value.
    - `angle_p`: probability that shuffled angle differences are smaller than the actual difference.
* Classify cells as HD when `rayleigh_p <= RAYLEIGH_P_THRESHOLD`, `angle_p <= ANGLE_P_THRESHOLD`,
  and at least one compartment has a significant Rayleigh vector.
* Summarize HD counts per mouse, plot stacked bars, and overlay per-session scatter points with robust medians.

This approach avoids hard-coded Rayleigh/angle thresholds while keeping results statistically grounded.
"""
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from behave_analysis.process.session import get_experiment
from behave_analysis.utils.rayleigh.load_rayleigh import extract_rayleigh_path, load_rayleigh_data
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr, JAL7_30apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may

experiments_objects = [JAL6_flip7_1apr, JAL6_flip3_18mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_28mar,
                       JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept,
                       JAL005_8thSept, JAL005_21stSept,
                       JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr,
                       JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip4_10may, JAL8_14may,
                       JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept]

session_names = ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar",
                 "JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept", 
                 "JAL5_8thSept", "JAL5_21stSept",
                 "JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr",
                 "JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may",
                 "JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"]

tinny_barrier = [JAL8_tiny_3may, JAL8_21may, JAL7_30apr]

# Mice groups based on session names
mice_groups = {
    "JAL6": ['JAL6_flip7_1apr', 'JAL6_flip3_18mar', 'JAL6_flip4_21mar', 'JAL6_flip5_25mar', 'JAL6_28mar'],
    "JAL3": ['JAL3_25aug', 'JAL3_1sept', 'JAL3_4sept', 'JAL3_7sept'],
    "JAL7": ['JAL7_sesh8_9apr', 'JAL7_sesh9_16apr', 'JAL7_flip5_22mar', 'JAL7_flip2_12mar', 'JAL7_23apr'],
    "JAL8": ['JAL8_flip1_25apr', 'JAL8_flip2_29apr', 'JAL8_flip4_10may', 'JAL8_14may'],
    "JAL4": ['JAL4_3rdSept', 'JAL4_19thSept', 'JAL4_28aug', 'JAL4_11thSept'],
    "JAL5": ['JAL5_8thSept', 'JAL5_21stSept']}

# Thresholds / parameters
CLUSTER_TYPE = "good"  # which set of clusters to load from disk
RAYLEIGH_P_THRESHOLD = 0.05  # max Rayleigh permutation p-value to call HD
ANGLE_P_THRESHOLD = 0.05  # max angle-difference permutation p-value to call HD
N_PERMUTATIONS = 1000  # number of shuffles for the null distribution
TINNY_SESSION_NAMES = {"JAL8_tiny_3may", "JAL8_21may", "JAL7_30apr"}  # sessions to skip
GLOBAL_RNG = np.random.default_rng(2024)  # reproducible RNG for permutations
CELL_CLASSIFICATION_SAVE_DIR = r"Z:\Laurence\thesis\cell_classification"
CELL_CLASSIFICATION_FILENAME = "head_direction_cells.json"


def circular_difference(theta1, theta2):
    """Return absolute circular difference between two angles (radians)."""
    return np.abs(np.angle(np.exp(1j * (theta1 - theta2))))

def permutation_pvalues(shelter_mag, barrier_mag, shelter_theta, barrier_theta, n_perm=N_PERMUTATIONS, rng=None):
    """Compute permutation-based p-values for Rayleigh magnitude similarity and angle stability."""
    if rng is None:
        rng = np.random.default_rng()

    shelter_mag = np.asarray(shelter_mag, dtype=float)
    barrier_mag = np.asarray(barrier_mag, dtype=float)
    shelter_theta = np.asarray(shelter_theta, dtype=float)
    barrier_theta = np.asarray(barrier_theta, dtype=float)
    n_cells = shelter_mag.shape[0]

    perm_rayleigh = np.empty((n_perm, n_cells), dtype=float)
    perm_angles = np.empty((n_perm, n_cells), dtype=float)

    for i in range(n_perm):
        perm_idx = rng.permutation(n_cells)
        perm_rayleigh[i] = np.sqrt(np.clip(shelter_mag * barrier_mag[perm_idx], a_min=0, a_max=None))
        perm_angles[i] = circular_difference(shelter_theta, barrier_theta[perm_idx])

    actual_rayleigh = np.sqrt(np.clip(shelter_mag * barrier_mag, a_min=0, a_max=None))
    actual_angle = circular_difference(shelter_theta, barrier_theta)

    rayleigh_p = (np.sum(perm_rayleigh >= actual_rayleigh, axis=0) + 1) / (n_perm + 1)
    angle_p = (np.sum(perm_angles <= actual_angle, axis=0) + 1) / (n_perm + 1)
    return rayleigh_p, angle_p, actual_rayleigh, actual_angle

def robust_summary(values):
    """Return the median and scaled MAD for a set of values (robust central tendency/spread)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, 1.4826 * mad

def canonicalize_mouse(label: str) -> str:
    """Normalize mouse labels by removing leading zeros (e.g., JAL006 -> JAL6)."""
    m = re.search(r"(JAL)0*([0-9]+)", label)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return label

def infer_mouse(session, session_to_mouse):
    """Infer mouse name for a session, using lookup table or session metadata."""
    # 1) by exact session name mapping
    session_name = getattr(session, "name", "")
    if session_name in session_to_mouse:
        return session_to_mouse[session_name]
    # 2) via attributes
    for attr in ("mouse", "mouse_id", "mouse_name"):
        val = getattr(session, attr, None)
        if isinstance(val, str) and val:
            return canonicalize_mouse(val)
    # 3) parse string repr "Mouse: JAL006, ..."
    m = re.search(r"Mouse:\s*(JAL0*\d+|JAL\d+)", str(session))
    if m:
        return canonicalize_mouse(m.group(1))
    return "UNKNOWN"

# ---- NEW: cross-backend column accessor (Polars/Pandas/iterable/dict) ----
def _to_list_like(col):
    if hasattr(col, "to_list"):   # Polars Series
        return col.to_list()
    if hasattr(col, "tolist"):    # NumPy/Pandas Series
        return col.tolist()
    try:
        return list(col)
    except Exception:
        return []

def column_as_list(df, name, default=None):
    """
    Return column values as a Python list across Polars/Pandas/dict.
    """
    if default is None:
        default = []
    if df is None:
        return default
    # dict-like
    if isinstance(df, dict):
        return _to_list_like(df.get(name, default))
    # DataFrame-like
    cols = getattr(df, "columns", None)
    if cols is not None and name in cols:
        try:
            col = df[name]
            return _to_list_like(col)
        except Exception:
            return default
    return default
# -------------------------------------------------------------------------

# Reverse lookup: session_name -> mouse (from your mice_groups)
session_to_mouse = {s: m for m, sessions in mice_groups.items() for s in sessions}

mouse_counts = defaultdict(lambda: {"total": 0, "hdir": 0})
session_level_counts = defaultdict(list)
session_hdir_map = {}
session_hd_mags = []
skipped = []

for idx, (session_obj, session_name) in enumerate(zip(experiments_objects, session_names)):
    session = get_experiment(session_obj)
    mouse = infer_mouse(session, session_to_mouse)
    save_session_name = session_names[idx]

    if save_session_name in TINNY_SESSION_NAMES:
        print(f"[INFO] Skipping {save_session_name}: tinny barrier session")
        skipped.append((mouse, save_session_name, "tinny barrier"))
        continue

    # Load Rayleigh data for all_time head direction per session
    try:
        path = extract_rayleigh_path(session, CLUSTER_TYPE, condition="shelter_only", file_name="hdir_Rayleigh.arrow")
        data = load_rayleigh_data(path)
    except Exception as e:
        msg = str(e)
        print(f"[WARN] Skipping {save_session_name}: {msg}")
        skipped.append((mouse, save_session_name, msg))
        continue
    
    print(data)

    cluster_ids = column_as_list(data, "clusterID", default=[])
    n_total = len(cluster_ids)

    if n_total == 0:
        print(f"[WARN] No clusters in {save_session_name}")
        skipped.append((mouse, save_session_name, "no clusters"))
        continue

    try:
        angles = np.asarray(extract_compartment_values(data, "Rayleigh_theta"), dtype=float)
        mags = np.asarray(extract_compartment_values(data, "Rayleigh"), dtype=float)
        sigs = np.asarray(extract_compartment_values(data, "Rayleigh_sig"), dtype=int)
    except Exception as exc:
        msg = f"failed to parse Rayleigh arrays ({exc})"
        print(f"[WARN] {save_session_name}: {msg}")
        skipped.append((mouse, save_session_name, msg))
        continue

    valid_mask = np.isfinite(mags).all(axis=1) & np.isfinite(angles).all(axis=1)
    sig_any_mask = (sigs[:, 0] > 0) | (sigs[:, 1] > 0)
    candidate_mask = valid_mask  # include all valid cells, even if not significant
    candidate_count = int(np.sum(candidate_mask))
    significant_count = int(np.sum(sig_any_mask & valid_mask))

    session_hdir_ids = []
    if candidate_count >= 2:
        shelter_mag = mags[candidate_mask, 0]
        barrier_mag = mags[candidate_mask, 1]
        shelter_theta = angles[candidate_mask, 0]
        barrier_theta = angles[candidate_mask, 1]

        rayleigh_p, angle_p, actual_rayleigh, _ = permutation_pvalues(
            shelter_mag, barrier_mag, shelter_theta, barrier_theta, n_perm=N_PERMUTATIONS, rng=GLOBAL_RNG
        )
        candidate_ids = np.asarray(cluster_ids)[candidate_mask]

        # Penalize cells that are not significant in either compartment
        candidate_sig_any = sig_any_mask[candidate_mask]
        angle_p = np.where(candidate_sig_any, angle_p, np.minimum(1.0, angle_p))

        keep_mask = (rayleigh_p <= RAYLEIGH_P_THRESHOLD) & (angle_p <= ANGLE_P_THRESHOLD) & candidate_sig_any
        session_hdir_ids = candidate_ids[keep_mask].tolist()
        session_hd_mags.extend(actual_rayleigh[keep_mask])
        session_hd_mags.extend(actual_rayleigh[keep_mask])
    elif candidate_count == 1:
        print(f"[WARN] {save_session_name}: only one valid cell, unable to run permutation test")
        skipped.append((mouse, save_session_name, "only one valid cell"))
    else:
        print(f"[INFO] {save_session_name}: no valid cells, treating as 0 head-direction cells")

    n_hdir = len(session_hdir_ids)
    mouse_counts[mouse]["total"] += n_total
    mouse_counts[mouse]["hdir"] += n_hdir
    session_level_counts[mouse].append({"session": save_session_name, "count": n_hdir, "total": n_total})
    session_hdir_map[save_session_name] = session_hdir_ids

    print(f"{save_session_name} ({mouse}): HDir {n_hdir} / {n_total} (candidates {candidate_count}, sig {significant_count})")

output_path = os.path.join(CELL_CLASSIFICATION_SAVE_DIR, CELL_CLASSIFICATION_FILENAME)
try:
    os.makedirs(CELL_CLASSIFICATION_SAVE_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_hdir_map, f, indent=2)
    print(f"[INFO] Saved head-direction cell IDs to {output_path}")
except Exception as exc:
    print(f"[WARN] Failed to save head-direction cell IDs to {output_path}: {exc}")

# Summary of skipped sessions
if skipped:
    by_reason = defaultdict(int)
    for _, _, msg in skipped:
        by_reason[msg.split(':')[0]] += 1
    print(f"\n[INFO] Skipped {len(skipped)} sessions. Top reasons:")
    for reason, cnt in sorted(by_reason.items(), key=lambda x: -x[1])[:5]:
        print(f"  {reason}: {cnt}")

# Prepare plot data
mice = sorted([m for m in mouse_counts.keys() if m != "UNKNOWN"]) or ["UNKNOWN"]
totals = np.array([mouse_counts[m]["total"] for m in mice], dtype=int)
hdirs = np.array([mouse_counts[m]["hdir"] for m in mice], dtype=int)
non_hdirs = totals - hdirs

# Plot stacked bars: HDir vs Non-HDir per mouse (counts)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(mice))
ax.bar(x, non_hdirs, color="#d0d0d0", label="Non-HDir")
ax.bar(x, hdirs, bottom=non_hdirs, color="#2ca02c", label="HDir")
ax.set_ylabel("Cells (across sessions)")

# Secondary axis for per-session HD percentages
ax2 = ax.twinx()
ax2.set_ylabel("Head-direction cells per session (%)")
ax2.set_ylim(0, 105)

# Overlay per-session crosses and robust medians on secondary axis
session_scatter_added = False
median_handle_added = False
for idx, mouse in enumerate(mice):
    entries = session_level_counts.get(mouse, [])
    if not entries:
        continue
    session_counts = np.array([entry["count"] for entry in entries], dtype=float)
    session_totals = np.array([entry.get("total", 0) for entry in entries], dtype=float)
    valid_sessions = session_totals > 0
    if not valid_sessions.any():
        continue
    session_counts = session_counts[valid_sessions]
    session_totals = session_totals[valid_sessions]
    session_pct = np.divide(session_counts, session_totals, out=np.zeros_like(session_counts), where=session_totals > 0) * 100.0
    x_positions = np.full_like(session_pct, fill_value=idx, dtype=float)
    scatter_label = "Session % HD" if not session_scatter_added else "_nolegend_"
    ax2.scatter(x_positions, session_pct, marker="x", color="k", s=50, alpha=0.8, label=scatter_label)
    session_scatter_added = True

    median, mad = robust_summary(session_pct)
    error = mad if session_pct.size > 1 else 0.0
    error_label = "Median +/- MAD" if not median_handle_added else "_nolegend_"
    ax2.errorbar(idx, median, yerr=error, ecolor="k", lw=1.3, capsize=4, label=error_label)
    if not median_handle_added:
        median_handle_added = True

# Annotate counts and total percentages on top of bars
for i, (t, h) in enumerate(zip(totals, hdirs)):
    pct_label = (100.0 * h / t) if t > 0 else 0.0
    ax.text(i, t + max(1, totals.max()) * 0.01, f"{h}/{t} ({pct_label:.2f}%)", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(mice, rotation=0)
ax.set_title("Head-direction cells across mice (aggregated over sessions)")

handles, labels = ax.get_legend_handles_labels()
sec_handles, sec_labels = ax2.get_legend_handles_labels()
handles.extend(sec_handles)
labels.extend(sec_labels)
ax.legend(handles, labels, frameon=False, loc="upper left")
ax.set_ylim(0, max(1, totals.max()) * 1.15)

# limit the y right axes to 0-5%
ax2.set_ylim(0, 5)

# Plot histogram of Rayleigh magnitudes for classified head-direction cells
session_hd_mags = np.array(session_hd_mags, dtype=float)
save_path = r"Z:\Laurence\thesis\efizz_chapter"
if session_hd_mags.size > 0:
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax_hist.hist(session_hd_mags, bins=bins, color="#5DADE2", edgecolor="white", alpha=0.85)
    ax_hist.set_xlabel("Geometric mean Rayleigh magnitude")
    ax_hist.set_ylabel("Count of head-direction cells")
    ax_hist.set_title("Rayleigh strength of classified HD cells")
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/hdir_rayleigh_histogram.eps", format='eps', dpi=150)
    plt.show()
else:
    print("[INFO] No head-direction cells classified; skipping Rayleigh histogram.")

# save as eps
plt.savefig(f"{save_path}/hdir_cells_across_mice.eps", format='eps', dpi=150)

plt.tight_layout()
plt.show()
