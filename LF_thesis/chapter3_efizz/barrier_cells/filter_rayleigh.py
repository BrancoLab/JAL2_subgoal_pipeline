import ast
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------
# Paths / constants
# -----------------------
SAVE_ROOT = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/Top2_TunED")
RAYLEIGH_CSV = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/threat_dict_max_rayleigh_flat.csv")
TUNED_IDS_LONG = SAVE_ROOT / "tuned_ids_A_or_B_long.csv"

OUT_CSV = SAVE_ROOT / "linked_tuned_ids_to_rayleigh_long.csv"
OUT_PKL = SAVE_ROOT / "linked_tuned_ids_to_rayleigh.pkl"

# Keep threat-zone rows by default (set to None to keep both compartments)
COMPARTMENT_FILTER = "threat"

# A/B tuning maps to these angle labels in the flat Rayleigh table
A_ANGLE = "h_preflipbar_a"
B_ANGLE = "h_postflipbar_a"


def _parse_firing_rate_series(x):
    """Parse stringified arrays like '[1.0 2.0 ...]' into float array."""
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]
    if pd.isna(x):
        return np.asarray([], dtype=float)
    s = str(x).strip()
    if not s:
        return np.asarray([], dtype=float)
    try:
        # Try Python-literal parse first, for comma-separated forms.
        parsed = ast.literal_eval(s)
        arr = np.asarray(parsed, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        # Fallback for space-separated forms: "[1.0 2.0 3.0]"
        s = s.strip("[]")
        arr = np.fromstring(s, sep=" ", dtype=float)
        return arr[np.isfinite(arr)]


def load_input_tables():
    if not TUNED_IDS_LONG.exists():
        raise FileNotFoundError(f"Tuned ID table not found: {TUNED_IDS_LONG}")
    if not RAYLEIGH_CSV.exists():
        raise FileNotFoundError(f"Rayleigh flat CSV not found: {RAYLEIGH_CSV}")

    tuned = pd.read_csv(TUNED_IDS_LONG)
    ray = pd.read_csv(RAYLEIGH_CSV)
    return tuned, ray


def build_tuned_targets(tuned_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"session", "cluster_id", "condition", "A_only", "B_only"}
    missing = needed - set(tuned_df.columns)
    if missing:
        raise KeyError(f"Missing columns in tuned IDs table: {sorted(missing)}")

    t = tuned_df.copy()
    t["A_only"] = t["A_only"].astype(bool)
    t["B_only"] = t["B_only"].astype(bool)
    t = t[t["A_only"] | t["B_only"]].copy()
    if t.empty:
        return t

    t["tuned_label"] = np.where(t["A_only"], "A_only", "B_only")
    t["angle"] = np.where(t["A_only"], A_ANGLE, B_ANGLE)
    return t[["session", "cluster_id", "condition", "tuned_label", "angle"]]


def filter_rayleigh_table(ray_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"session", "cluster_id", "condition", "angle", "rayleigh", "firing_rate_hz"}
    missing = needed - set(ray_df.columns)
    if missing:
        raise KeyError(f"Missing columns in Rayleigh table: {sorted(missing)}")

    r = ray_df.copy()
    if COMPARTMENT_FILTER is not None and "compartment" in r.columns:
        r = r[r["compartment"].astype(str).str.lower() == COMPARTMENT_FILTER.lower()].copy()
    return r


def link_tuned_with_rayleigh(tuned_targets: pd.DataFrame, ray_df: pd.DataFrame) -> pd.DataFrame:
    merged = tuned_targets.merge(
        ray_df,
        on=["session", "cluster_id", "condition", "angle"],
        how="left",
        validate="one_to_many",
    )

    fr_arrays = merged["firing_rate_hz"].apply(_parse_firing_rate_series)
    merged["firing_rate_mean_hz"] = fr_arrays.apply(lambda a: float(np.mean(a)) if a.size else np.nan)
    merged["firing_rate_max_hz"] = fr_arrays.apply(lambda a: float(np.max(a)) if a.size else np.nan)
    return merged


def main():
    tuned_df, ray_df = load_input_tables()
    tuned_targets = build_tuned_targets(tuned_df)
    if tuned_targets.empty:
        raise RuntimeError("No A-only or B-only rows found in tuned ID table.")

    ray_filtered = filter_rayleigh_table(ray_df)
    out_df = link_tuned_with_rayleigh(tuned_targets, ray_filtered)

    out_df.to_csv(OUT_CSV, index=False)
    out_df.to_pickle(OUT_PKL)

    n_total = len(tuned_targets)
    n_hit = int(out_df["rayleigh"].notna().sum()) if "rayleigh" in out_df.columns else 0
    print(f"Input tuned rows (A or B only): {n_total}")
    print(f"Matched Rayleigh rows: {n_hit}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_PKL}")


if __name__ == "__main__":
    main()
