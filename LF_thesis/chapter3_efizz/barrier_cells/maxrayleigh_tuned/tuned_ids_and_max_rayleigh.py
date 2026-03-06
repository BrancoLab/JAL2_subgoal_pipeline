import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------
# Paths / constants
# -----------------------
SAVE_ROOT = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/Top2_TunED")
RAYLEIGH_CSV = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/threat_dict_max_rayleigh_flat.csv")

PICKLE_AB = SAVE_ROOT / "A_vs_B_all_conditions_threat_zone.pkl"
PICKLE_A_HSA = SAVE_ROOT / "A_vs_HSA.pkl"
PICKLE_B_HSA = SAVE_ROOT / "B_vs_HSA.pkl"

OUT_LONG = SAVE_ROOT / "tuned_ids_and_max_rayleigh_long.csv"
OUT_A = SAVE_ROOT / "tuned_ids_A_preflip_AND_maxrayleighA.csv"
OUT_B = SAVE_ROOT / "tuned_ids_B_postflip_AND_maxrayleighB.csv"

CONDITION_PRE = "barrier_pre_flip"
CONDITION_POST = "barrier_post_flip"
A_ANGLE = "h_preflipbar_a"
B_ANGLE = "h_postflipbar_a"
COMPARTMENT_FILTER = "threat"


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected pickle structure in {path}")
    return data


def session_to_mouse(session: str) -> str:
    m = re.match(r"^(JAL)(\d+)", str(session))
    if not m:
        return "UNKNOWN"
    return f"JAL{int(m.group(2))}"


def allowed_ids_from_hsa(results: dict, tuning_key: str) -> dict:
    """
    Return ids that are not HSA-only (allow mixed cells).
    Output shape: allowed[session][condition] = set(cluster_ids)
    """
    allowed = {}
    for session, cond_dict in results.items():
        allowed_session = {}
        for cond, clusters in (cond_dict or {}).items():
            allowed_ids = set()
            if isinstance(clusters, dict):
                for cluster_id, flags in clusters.items():
                    mixed = bool(flags.get("mixed_tuning", False))
                    target_tuned = bool(flags.get(tuning_key, False))
                    hsa_tuned = bool(flags.get("hsa_tuned", False))
                    if mixed or target_tuned or not hsa_tuned:
                        allowed_ids.add(cluster_id)
            allowed_session[cond] = allowed_ids
        allowed[session] = allowed_session
    return allowed


def build_tuned_ab_table(results_ab: dict, allowed_pre: dict, allowed_post: dict) -> pd.DataFrame:
    """
    Build per-session/condition cluster tuning labels from the A-vs-B model:
      A_only = preflipbar-A tuned AND not postflipbar-A tuned
      B_only = postflipbar-A tuned AND not preflipbar-A tuned
    """
    rows = []
    for session, cond_dict in results_ab.items():
        for cond in [CONDITION_PRE, CONDITION_POST]:
            clusters = (cond_dict or {}).get(cond, {}) or {}
            if not isinstance(clusters, dict):
                continue

            pre_allowed = allowed_pre.get(session, {}).get(cond, set())
            post_allowed = allowed_post.get(session, {}).get(cond, set())

            for cluster_id, flags in clusters.items():
                pre_tuned = bool(flags.get("h_preflipbar_a_tuned", False)) and (cluster_id in pre_allowed)
                post_tuned = bool(flags.get("h_postflipbar_a_tuned", False)) and (cluster_id in post_allowed)
                a_only = pre_tuned and not post_tuned
                b_only = post_tuned and not pre_tuned

                rows.append(
                    {
                        "session": session,
                        "condition": cond,
                        "cluster_id": cluster_id,
                        "A_only": a_only,
                        "B_only": b_only,
                    }
                )
    return pd.DataFrame(rows)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of expected columns found: {candidates}")


def build_max_rayleigh_table(ray_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (session, condition, cluster_id), keep the angle row with max rayleigh.
    """
    session_col = _find_col(ray_df, ["session"])
    condition_col = _find_col(ray_df, ["condition"])
    cluster_col = _find_col(ray_df, ["cluster_id", "cluster"])
    angle_col = _find_col(ray_df, ["angle"])
    rayleigh_col = _find_col(ray_df, ["rayleigh"])

    r = ray_df.rename(
        columns={
            session_col: "session",
            condition_col: "condition",
            cluster_col: "cluster_id",
            angle_col: "angle",
            rayleigh_col: "rayleigh",
        }
    ).copy()

    if COMPARTMENT_FILTER is not None and "compartment" in r.columns:
        r = r[r["compartment"].astype(str).str.lower() == COMPARTMENT_FILTER.lower()].copy()

    r["rayleigh"] = pd.to_numeric(r["rayleigh"], errors="coerce")
    r = r.dropna(subset=["rayleigh", "session", "condition", "cluster_id", "angle"])
    if r.empty:
        return pd.DataFrame(columns=["session", "condition", "cluster_id", "max_rayleigh_angle", "max_rayleigh_value"])

    # Deterministic tie-break: highest rayleigh, then lexicographic angle.
    r = r.sort_values(["session", "condition", "cluster_id", "rayleigh", "angle"], ascending=[True, True, True, False, True])
    top = r.drop_duplicates(subset=["session", "condition", "cluster_id"], keep="first").copy()
    top = top.rename(columns={"angle": "max_rayleigh_angle", "rayleigh": "max_rayleigh_value"})
    return top[["session", "condition", "cluster_id", "max_rayleigh_angle", "max_rayleigh_value"]]


def apply_and_logic(tuned_df: pd.DataFrame, max_ray_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that satisfy:
      A accepted if:
        - condition == barrier_pre_flip
        - A_only is True
        - max-rayleigh angle == h_preflipbar_a
      B accepted if:
        - condition == barrier_post_flip
        - B_only is True
        - max-rayleigh angle == h_postflipbar_a
    """
    merged = tuned_df.merge(
        max_ray_df,
        on=["session", "condition", "cluster_id"],
        how="left",
        validate="many_to_one",
    )

    merged["max_rayleigh_angle"] = merged["max_rayleigh_angle"].astype(str)

    merged["A_and_maxA"] = (
        (merged["condition"] == CONDITION_PRE)
        & merged["A_only"].astype(bool)
        & (merged["max_rayleigh_angle"] == A_ANGLE)
    )
    merged["B_and_maxB"] = (
        (merged["condition"] == CONDITION_POST)
        & merged["B_only"].astype(bool)
        & (merged["max_rayleigh_angle"] == B_ANGLE)
    )

    merged["mouse"] = merged["session"].map(session_to_mouse)
    return merged


def main():
    print("Loading tuned-model pickles...")
    results_ab = load_pickle(PICKLE_AB)
    results_a_hsa = load_pickle(PICKLE_A_HSA)
    results_b_hsa = load_pickle(PICKLE_B_HSA)

    print("Building tuned allow-lists...")
    allowed_pre = allowed_ids_from_hsa(results_a_hsa, "h_preflipbar_a_tuned")
    allowed_post = allowed_ids_from_hsa(results_b_hsa, "h_postflipbar_a_tuned")
    tuned_df = build_tuned_ab_table(results_ab, allowed_pre, allowed_post)
    if tuned_df.empty:
        raise RuntimeError("No tuned rows built from A-vs-B pickle.")

    print("Loading Rayleigh table and finding max-rayleigh angle per cell...")
    if not RAYLEIGH_CSV.exists():
        raise FileNotFoundError(f"Rayleigh flat CSV not found: {RAYLEIGH_CSV}")
    ray_df = pd.read_csv(RAYLEIGH_CSV)
    max_ray_df = build_max_rayleigh_table(ray_df)
    if max_ray_df.empty:
        raise RuntimeError("No usable rows found in Rayleigh table after filtering.")

    print("Applying AND logic (tuned label AND expected max-rayleigh angle)...")
    out_df = apply_and_logic(tuned_df, max_ray_df)
    out_df.to_csv(OUT_LONG, index=False)

    a_df = out_df[out_df["A_and_maxA"]].copy()
    b_df = out_df[out_df["B_and_maxB"]].copy()

    a_df[["mouse", "session", "condition", "cluster_id", "max_rayleigh_angle", "max_rayleigh_value"]].to_csv(OUT_A, index=False)
    b_df[["mouse", "session", "condition", "cluster_id", "max_rayleigh_angle", "max_rayleigh_value"]].to_csv(OUT_B, index=False)

    print(f"Saved long audit table: {OUT_LONG}")
    print(f"Saved A accepted IDs:    {OUT_A}")
    print(f"Saved B accepted IDs:    {OUT_B}")

    summary = (
        out_df.groupby(["mouse", "session", "condition"], as_index=False)
        .agg(
            n_A_only=("A_only", "sum"),
            n_B_only=("B_only", "sum"),
            n_A_and_maxA=("A_and_maxA", "sum"),
            n_B_and_maxB=("B_and_maxB", "sum"),
        )
        .sort_values(["mouse", "session", "condition"])
    )
    print("\nCounts by mouse/session/condition:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
