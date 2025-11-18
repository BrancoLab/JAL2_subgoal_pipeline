"""Head-shelter angle (HSA) classification using permutation logic shared with head-direction analysis.

Logic summary:
1. Load precomputed Rayleigh data (magnitudes, preferred angles, and significance flags) for each session.
2. For every cluster with valid values, build permutation distributions by shuffling shelter/threat pairings.
3. Use geometric-mean Rayleigh magnitude and circular angle difference as test statistics.
4. Convert those distributions into per-cell p-values:
   - `rayleigh_p`: chance shuffle magnitudes exceed the observed value.
   - `angle_p`: chance shuffle angle differences are smaller than the observed difference.
5. Label an HSA cell when `rayleigh_p <= RAYLEIGH_P_THRESHOLD`, `angle_p <= ANGLE_P_THRESHOLD`,
   at least one compartment is significant, and the cluster is not already labelled as head-direction.
6. Aggregate counts per mouse, visualize stacked totals plus per-session scatter, and display Rayleigh histograms.
"""

import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_11thSept, JAL4_19thSept, JAL4_28aug, JAL4_3rdSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_21stSept, JAL005_8thSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip3_18mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_23apr, JAL7_30apr, JAL7_flip2_12mar, JAL7_flip5_22mar, JAL7_sesh8_9apr, JAL7_sesh9_16apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_14may, JAL8_21may, JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip4_10may, JAL8_tiny_3may
from behave_analysis.process.session import get_experiment
from behave_analysis.utils.rayleigh.load_rayleigh import extract_rayleigh_path, load_rayleigh_data
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values

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

session_names = [
    "JAL6_flip7_1apr",
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL5_8thSept",
    "JAL5_21stSept",
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

TINNY_SESSION_NAMES = {"JAL8_tiny_3may", "JAL8_21may", "JAL7_30apr"}

mice_groups = {
    "JAL6": ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may"],
    "JAL4": ["JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"],
    "JAL5": ["JAL5_8thSept", "JAL5_21stSept"],
}

CLUSTER_TYPE = "good"
RAYLEIGH_P_THRESHOLD = 0.05
ANGLE_P_THRESHOLD = 0.05
MIN_HSA_MAG = 0.2  # ensure HSA tuning is reasonably strong
DOMINANCE_MARGIN = 0.05  # HSA magnitude must exceed other angles by this margin
N_PERMUTATIONS = 1000
GLOBAL_RNG = np.random.default_rng(2024)
CELL_CLASSIFICATION_SAVE_DIR = r"Z:\Laurence\thesis\cell_classification"
CELL_CLASSIFICATION_FILENAME = "head_shelter_cells.json"
HD_EXCLUSION_PATH = r"Z:\Laurence\thesis\cell_classification\head_direction_cells.json"

ANGLE_FILES = {
    "hsa": "hsa_Rayleigh.arrow",
    "preflip": "h_preflipbar_a_Rayleigh.arrow",
    "postflip": "h_postflipbar_a_Rayleigh.arrow",
    "bar_centre": "h_bar_centre_a_Rayleigh.arrow",
}


def circular_difference(theta1, theta2):
    """Return the absolute circular difference between two angles (radians)."""
    return np.abs(np.angle(np.exp(1j * (theta1 - theta2))))


def permutation_pvalues(shelter_mag, barrier_mag, shelter_theta, barrier_theta, n_perm=N_PERMUTATIONS, rng=None):
    """Permutation-based p-values for geometric-mean Rayleigh magnitude and angle stability."""
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
    """Return the median and scaled MAD for a sequence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, 1.4826 * mad


def canonicalize_mouse(label):
    """Convert JAL006 -> JAL6, etc."""
    m = re.search(r"(JAL)0*([0-9]+)", label or "")
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return label


def infer_mouse(session, session_to_mouse):
    """Infer the mouse label for a session."""
    session_name = getattr(session, "name", "")
    if session_name in session_to_mouse:
        return session_to_mouse[session_name]
    for attr in ("mouse", "mouse_id", "mouse_name"):
        text = getattr(session, attr, None)
        if isinstance(text, str) and text:
            return canonicalize_mouse(text)
    found = re.search(r"Mouse:\s*(JAL0*\d+|JAL\d+)", str(session))
    if found:
        return canonicalize_mouse(found.group(1))
    return "UNKNOWN"


def _to_list_like(col):
    if hasattr(col, "to_list"):
        return col.to_list()
    if hasattr(col, "tolist"):
        return col.tolist()
    try:
        return list(col)
    except Exception:
        return []


def column_as_list(df, name, default=None):
    """Return a column as list for polars/pandas/dicts."""
    if default is None:
        default = []
    if df is None:
        return default
    if isinstance(df, dict):
        return _to_list_like(df.get(name, default))
    cols = getattr(df, "columns", None)
    if cols is not None and name in cols:
        try:
            col = df[name]
            return _to_list_like(col)
        except Exception:
            return default
    return default


def load_hdir_exclusions(path=HD_EXCLUSION_PATH):
    """Load HD classifications that should be excluded from HSA."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: set(v) for k, v in data.items()}
    except Exception as exc:
        print(f"[WARN] Failed to load HD exclusion list ({exc}). Continuing without exclusions.")
        return {}


def load_angle_dataframe(session, file_name):
    """Load a Rayleigh arrow for a specific angle."""
    path = extract_rayleigh_path(session, CLUSTER_TYPE, condition="shelter_only", file_name=file_name)
    return load_rayleigh_data(path)


def align_metrics(reference_ids, df):
    """Align rayleigh magnitudes/significance to the reference cluster order."""
    if df is None:
        return np.zeros((len(reference_ids), 2), dtype=float), np.zeros((len(reference_ids), 2), dtype=int)
    ids = column_as_list(df, "clusterID", default=[])
    mag = np.asarray(extract_compartment_values(df, "Rayleigh"), dtype=float)
    sig = np.asarray(extract_compartment_values(df, "Rayleigh_sig"), dtype=int)
    id_to_idx = {cid: idx for idx, cid in enumerate(ids)}
    aligned_mag = np.zeros((len(reference_ids), 2), dtype=float)
    aligned_sig = np.zeros((len(reference_ids), 2), dtype=int)
    for out_idx, ref in enumerate(reference_ids):
        match_idx = id_to_idx.get(ref)
        if match_idx is not None:
            aligned_mag[out_idx] = mag[match_idx]
            aligned_sig[out_idx] = sig[match_idx]
    return aligned_mag, aligned_sig


def geomean_mag(mags):
    mags = np.asarray(mags, dtype=float)
    return np.sqrt(np.clip(mags[:, 0] * mags[:, 1], a_min=0, a_max=None))


session_to_mouse = {name: mouse for mouse, sessions in mice_groups.items() for name in sessions}
hd_exclusions = load_hdir_exclusions()

mouse_counts = defaultdict(lambda: {"total": 0, "hsa": 0})
session_level_counts = defaultdict(list)
session_hsa_map = {}
session_hsa_mags = []
skipped = []

for session_obj, session_name in zip(experiments_objects, session_names):
    session = get_experiment(session_obj)
    mouse = infer_mouse(session, session_to_mouse)

    if session_name in TINNY_SESSION_NAMES:
        print(f"[INFO] Skipping {session_name}: tinny barrier session")
        skipped.append((mouse, session_name, "tinny barrier"))
        continue

    angle_data = {}
    missing_critical = False
    for key, fname in ANGLE_FILES.items():
        try:
            angle_data[key] = load_angle_dataframe(session, fname)
        except Exception as exc:
            if key == "hsa":
                missing_critical = True
                warn = f"missing required HSA file ({exc})"
                break
            else:
                print(f"[WARN] {session_name}: missing {fname} ({exc}), assuming zeros")
                angle_data[key] = None
    if missing_critical:
        print(f"[WARN] Skipping {session_name}: {warn}")
        skipped.append((mouse, session_name, warn))
        continue

    data = angle_data["hsa"]
    cluster_ids = column_as_list(data, "clusterID", default=[])
    n_total = len(cluster_ids)
    if n_total == 0:
        print(f"[WARN] No clusters in {session_name}")
        skipped.append((mouse, session_name, "no clusters"))
        continue

    try:
        angles = np.asarray(extract_compartment_values(data, "Rayleigh_theta"), dtype=float)
        mags = np.asarray(extract_compartment_values(data, "Rayleigh"), dtype=float)
        sigs = np.asarray(extract_compartment_values(data, "Rayleigh_sig"), dtype=int)
    except Exception as exc:
        print(f"[WARN] {session_name}: failed to parse Rayleigh arrays ({exc})")
        skipped.append((mouse, session_name, "parse failure"))
        continue

    valid_mask = np.isfinite(mags).all(axis=1) & np.isfinite(angles).all(axis=1)
    sig_any_mask = (sigs[:, 0] > 0) | (sigs[:, 1] > 0)
    candidate_mask = valid_mask
    candidate_count = int(np.sum(candidate_mask))
    session_hsa_ids = []

    aligned_other = {}
    for other_key in ("preflip", "postflip", "bar_centre"):
        if other_key in angle_data:
            mag_other, sig_other = align_metrics(cluster_ids, angle_data[other_key])
        else:
            mag_other = np.zeros((len(cluster_ids), 2))
            sig_other = np.zeros((len(cluster_ids), 2))
        aligned_other[other_key] = {
            "mag": mag_other,
            "sig_any": (sig_other[:, 0] > 0) | (sig_other[:, 1] > 0),
        }

    if candidate_count >= 2:
        shelter_mag = mags[candidate_mask, 0]
        barrier_mag = mags[candidate_mask, 1]
        shelter_theta = angles[candidate_mask, 0]
        barrier_theta = angles[candidate_mask, 1]

        rayleigh_p, angle_p, actual_rayleigh, _ = permutation_pvalues(
            shelter_mag, barrier_mag, shelter_theta, barrier_theta, n_perm=N_PERMUTATIONS, rng=GLOBAL_RNG
        )
        candidate_ids = np.asarray(cluster_ids)[candidate_mask]
        candidate_sig_any = sig_any_mask[candidate_mask]
        angle_p = np.where(candidate_sig_any, angle_p, np.minimum(1.0, angle_p))

        keep_mask = (rayleigh_p <= RAYLEIGH_P_THRESHOLD) & (angle_p <= ANGLE_P_THRESHOLD) & candidate_sig_any

        hsa_geomean = geomean_mag(mags[candidate_mask])
        dominance_mask = hsa_geomean >= MIN_HSA_MAG
        for other_key, metrics in aligned_other.items():
            other_geo = geomean_mag(metrics["mag"][candidate_mask])
            other_sig = metrics["sig_any"][candidate_mask]
            dominance_mask &= (~other_sig) | (hsa_geomean >= other_geo + DOMINANCE_MARGIN)

        keep_mask &= dominance_mask

        exclusion_ids = hd_exclusions.get(session_name, set())
        if exclusion_ids:
            keep_mask &= ~np.isin(candidate_ids, list(exclusion_ids))

        session_hsa_ids = candidate_ids[keep_mask].tolist()
        session_hsa_mags.extend(hsa_geomean[keep_mask])
    elif candidate_count == 1:
        print(f"[WARN] {session_name}: only one valid cell, unable to run permutation test")
        skipped.append((mouse, session_name, "single candidate"))
    else:
        print(f"[INFO] {session_name}: no valid cells, treating as 0 head-shelter cells")

    n_hsa = len(session_hsa_ids)
    mouse_counts[mouse]["total"] += n_total
    mouse_counts[mouse]["hsa"] += n_hsa
    session_level_counts[mouse].append({"session": session_name, "count": n_hsa, "total": n_total})
    session_hsa_map[session_name] = session_hsa_ids

    print(f"{session_name} ({mouse}): HSA {n_hsa} / {n_total} (candidates {candidate_count})")

output_path = os.path.join(CELL_CLASSIFICATION_SAVE_DIR, CELL_CLASSIFICATION_FILENAME)
try:
    os.makedirs(CELL_CLASSIFICATION_SAVE_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(session_hsa_map, fh, indent=2)
    print(f"[INFO] Saved head-shelter cell IDs to {output_path}")
except Exception as exc:
    print(f"[WARN] Failed to save head-shelter cell IDs: {exc}")

if skipped:
    summary = defaultdict(int)
    for _, _, msg in skipped:
        summary[msg.split(":")[0]] += 1
    print(f"\n[INFO] Skipped {len(skipped)} sessions.")
    for reason, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

mice = sorted([m for m in mouse_counts if m != "UNKNOWN"]) or ["UNKNOWN"]
totals = np.array([mouse_counts[m]["total"] for m in mice], dtype=int)
hsas = np.array([mouse_counts[m]["hsa"] for m in mice], dtype=int)
non_hsa = totals - hsas

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(mice))
ax.bar(x, non_hsa, color="#d0d0d0", label="Non-HSA")
ax.bar(x, hsas, bottom=non_hsa, color="#c0392b", label="HSA")
ax.set_ylabel("Cells (across sessions)")

ax2 = ax.twinx()
ax2.set_ylabel("Head-shelter cells per session (%)")
ax2.set_ylim(0, 5)

session_scatter_added = False
median_handle_added = False
for idx, mouse in enumerate(mice):
    entries = session_level_counts.get(mouse, [])
    if not entries:
        continue
    session_counts = np.array([entry["count"] for entry in entries], dtype=float)
    session_totals = np.array([entry["total"] for entry in entries], dtype=float)
    valid = session_totals > 0
    if not valid.any():
        continue
    session_counts = session_counts[valid]
    session_totals = session_totals[valid]
    session_pct = np.divide(session_counts, session_totals, out=np.zeros_like(session_counts), where=session_totals > 0) * 100.0
    x_positions = np.full_like(session_pct, fill_value=idx, dtype=float)
    scatter_label = "Session % HSA" if not session_scatter_added else "_nolegend_"
    ax2.scatter(x_positions, session_pct, marker="x", color="k", s=50, alpha=0.8, label=scatter_label)
    session_scatter_added = True

    median, mad = robust_summary(session_pct)
    error = mad if session_pct.size > 1 else 0.0
    error_label = "Median +/- MAD" if not median_handle_added else "_nolegend_"
    ax2.errorbar(idx, median, yerr=error, ecolor="k", lw=1.3, capsize=4, label=error_label)
    if not median_handle_added:
        median_handle_added = True

for i, (t, h) in enumerate(zip(totals, hsas)):
    pct_label = (100.0 * h / t) if t > 0 else 0.0
    ax.text(i, t + max(1, totals.max()) * 0.01, f"{h}/{t} ({pct_label:.2f}%)", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(mice, rotation=0)
ax.set_title("Head-shelter cells across mice (aggregated over sessions)")

handles, labels = ax.get_legend_handles_labels()
sec_handles, sec_labels = ax2.get_legend_handles_labels()
handles.extend(sec_handles)
labels.extend(sec_labels)
ax.legend(handles, labels, frameon=False, loc="upper left")
ax.set_ylim(0, max(1, totals.max()) * 1.15)

session_hsa_mags = np.array(session_hsa_mags, dtype=float)
save_path = r"Z:\Laurence\thesis\efizz_chapter"
plt.savefig(f"{save_path}/hsa_cells_across_mice.eps", format="eps", dpi=150)

if session_hsa_mags.size > 0:
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    ax_hist.hist(session_hsa_mags, bins=np.linspace(0, 1, 21), color="#e74c3c", edgecolor="white", alpha=0.85)
    ax_hist.set_xlabel("Geometric mean Rayleigh magnitude")
    ax_hist.set_ylabel("Count of head-shelter cells")
    ax_hist.set_title("Rayleigh strength of classified head-shelter cells")
    for spine in ("top", "right"):
        ax_hist.spines[spine].set_visible(False)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/hsa_rayleigh_histogram.eps", format="eps", dpi=150)
    plt.show()
else:
    print("[INFO] No head-shelter cells classified; skipping Rayleigh histogram.")

plt.tight_layout()