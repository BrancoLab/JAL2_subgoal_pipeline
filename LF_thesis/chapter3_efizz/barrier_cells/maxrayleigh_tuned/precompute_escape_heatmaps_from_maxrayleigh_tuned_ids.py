import argparse
import gc
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _short_month(mm: int) -> str:
    return ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][mm - 1]


def _mouse_num(name: str):
    m = re.search(r"JAL0*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"0*(\d+)_", name)
    if m:
        return int(m.group(1))
    return None


def _flip_num(name: str):
    m = re.search(r"flip[_ ]?(\d+)", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _day_month_from_target(name: str):
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?([A-Za-z]{3,9})", name, flags=re.IGNORECASE)
    if not m:
        return None
    dd = int(m.group(1))
    mon = m.group(2).lower()
    mm = MONTHS.get(mon)
    if mm is None:
        return None
    return dd, mm


def _day_month_from_discovered(name: str):
    m = re.search(r"(20\d{2})_(\d{2})_(\d{2})", name)
    if m:
        return int(m.group(3)), int(m.group(2))
    return _day_month_from_target(name)


def _target_alias_keys(name: str):
    mouse = _mouse_num(name)
    flip = _flip_num(name)
    dm = _day_month_from_target(name)
    keys = []
    if mouse is None:
        return keys
    if flip is not None and dm is not None:
        keys.append(f"m{mouse}|f{flip}|d{dm[0]}{_short_month(dm[1])}")
    if dm is not None:
        keys.append(f"m{mouse}|d{dm[0]}{_short_month(dm[1])}")
    if flip is not None:
        keys.append(f"m{mouse}|f{flip}")
    return keys


def _discovered_alias_keys(name: str):
    mouse = _mouse_num(name)
    flip = _flip_num(name)
    dm = _day_month_from_discovered(name)
    keys = []
    if mouse is None:
        return keys
    if flip is not None and dm is not None:
        keys.append(f"m{mouse}|f{flip}|d{dm[0]}{_short_month(dm[1])}")
    if dm is not None:
        keys.append(f"m{mouse}|d{dm[0]}{_short_month(dm[1])}")
    if flip is not None:
        keys.append(f"m{mouse}|f{flip}")
    return keys


def resolve_session_names(target_sessions, discovered_session_names):
    key_to_discovered = {}
    for ds in discovered_session_names:
        for key in _discovered_alias_keys(ds):
            key_to_discovered.setdefault(key, []).append(ds)

    mapping, unresolved, ambiguous = {}, [], {}
    for ts in sorted(set(target_sessions)):
        candidates = []
        for key in _target_alias_keys(ts):
            hits = key_to_discovered.get(key, [])
            if len(hits) == 1:
                mapping[ts] = hits[0]
                candidates = []
                break
            if len(hits) > 1:
                candidates = hits
        else:
            if candidates:
                ambiguous[ts] = sorted(set(candidates))
            else:
                unresolved.append(ts)
    return mapping, unresolved, ambiguous


def ensure_bool_columns(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col in ["shelter", "barrier_present", "barrier_flipped", "EscapePeriod", "OutofshelterIdx", "homingPeriod"]:
        if col in out.columns:
            out = out.with_columns(pl.col(col).cast(pl.Boolean))
    return out


def load_targets(a_csv: Path, b_csv: Path) -> pd.DataFrame:
    a = pd.read_csv(a_csv)[["session", "condition", "cluster_id"]].copy()
    b = pd.read_csv(b_csv)[["session", "condition", "cluster_id"]].copy()
    a["tuned_label"] = "A_only"
    b["tuned_label"] = "B_only"
    out = pd.concat([a, b], ignore_index=True)
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
    out = out.dropna(subset=["session", "cluster_id"]).copy()
    out["cluster_id"] = out["cluster_id"].astype(int)
    return out.drop_duplicates().sort_values(["session", "cluster_id", "condition"]).reset_index(drop=True)


def find_session_processed_dirs(root: Path) -> dict:
    out = {}
    for p in root.rglob("full_video_dataframe.csv"):
        if p.parent.name != "processed_data":
            continue
        out.setdefault(p.parent.parent.name, p.parent)
    return out


def _normalize_points(obj):
    """Convert nested coordinates to plain python int lists for json serialization."""
    if obj is None:
        return None
    out = []
    for pt in obj:
        if pt is None:
            continue
        out.append([int(pt[0]), int(pt[1])])
    return out


def load_arena_locations(processed_dir: Path):
    meta_path = processed_dir / "metadata"
    if not meta_path.exists():
        return None, None
    try:
        meta = pickle.load(open(meta_path, "rb"))
    except Exception:
        return None, None

    shelter = _normalize_points(getattr(meta, "shelter_location", None))
    barrier = _normalize_points(getattr(meta, "barrier_location", None))
    return shelter, barrier


def add_bin_edges(vpos_pd: pd.DataFrame, nbins: int):
    x_min, x_max = np.nanmin(vpos_pd["x"].values), np.nanmax(vpos_pd["x"].values)
    y_min, y_max = np.nanmin(vpos_pd["y"].values), np.nanmax(vpos_pd["y"].values)
    eps = 1e-9
    x_edges = np.linspace(x_min, x_max + eps, nbins + 1)
    y_edges = np.linspace(y_min, y_max + eps, nbins + 1)
    return x_edges, y_edges


def apply_bins(df: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray):
    nbins_x = len(x_edges) - 1
    nbins_y = len(y_edges) - 1
    df["x_bins"] = np.clip(np.digitize(df["x"].values, x_edges) - 1, 0, nbins_x - 1)
    df["y_bins"] = np.clip(np.digitize(df["y"].values, y_edges) - 1, 0, nbins_y - 1)


def extract_escape_bouts(vdf_raw: pl.DataFrame, condition: str):
    base = filter_video_dataframe(vdf_raw, condition, outofshelter=True, exclude_escape=False)
    if "EscapePeriod" not in base.columns:
        return []
    esc = base.filter(pl.col("EscapePeriod") == True)
    if esc.is_empty():
        return []
    frames = np.sort(esc["frames"].to_numpy().astype(int))
    if frames.size == 0:
        return []
    splits = np.where(np.diff(frames) > 1)[0] + 1
    return [chunk for chunk in np.split(frames, splits) if chunk.size > 0]


def load_session_data(processed_dir: Path, needed_clusters=None):
    video_csv = processed_dir / "full_video_dataframe.csv"
    spike_csv = processed_dir / "spike_count_by_frame_and_goodcluster.csv"
    if not video_csv.exists() or not spike_csv.exists():
        return None, None

    # Load only columns used downstream to reduce memory footprint.
    v_cols = pl.read_csv(video_csv, n_rows=0).columns
    keep_vcols = [
        c
        for c in [
            "frames",
            "mouse_x_position",
            "mouse_y_position",
            "OutofshelterIdx",
            "EscapePeriod",
            "shelter",
            "barrier_present",
            "barrier_flipped",
            "homingPeriod",
        ]
        if c in v_cols
    ]
    vdf_raw = ensure_bool_columns(pl.read_csv(video_csv, columns=keep_vcols))

    # Stream spike csv and keep only requested clusters when provided.
    scan = pl.scan_csv(spike_csv).select(["spike_aligned_to_frame", "spike_clusters", "spike_count"])
    if needed_clusters:
        cluster_vals = [int(x) for x in needed_clusters]
        scan = scan.filter(pl.col("spike_clusters").is_in(cluster_vals))
    sdf = (
        scan.collect(streaming=True)
        .rename({"spike_aligned_to_frame": "frame"})
        .with_columns(
            pl.col("frame").cast(pl.Int64),
            pl.col("spike_clusters").cast(pl.Int32),
            pl.col("spike_count").cast(pl.Float32),
        )
    )
    sdf_pd = sdf.to_pandas()
    return vdf_raw, sdf_pd


def build_rate_for_escape(base_df: pd.DataFrame, clu_spk: pd.DataFrame, fps: float, nbins: int):
    occ = base_df.groupby(["y_bins", "x_bins"]).size().unstack(fill_value=0)
    occ = occ.reindex(index=np.arange(nbins), columns=np.arange(nbins), fill_value=0)
    df_unit = base_df.merge(clu_spk, on="frame", how="left")
    df_unit["spike_count"] = df_unit["spike_count"].fillna(0)
    df_spk = df_unit[df_unit["spike_count"] > 0]
    if df_spk.empty:
        spk = pd.DataFrame(0, index=occ.index, columns=occ.columns)
    else:
        spk = df_spk.groupby(["y_bins", "x_bins"])["spike_count"].sum().unstack(fill_value=0)
        spk = spk.reindex(index=occ.index, columns=occ.columns, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = spk.values / np.where(occ.values == 0, np.nan, occ.values) * fps
    return rate, int(df_unit["spike_count"].sum()), int(len(df_unit))


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute per-escape heatmaps for max-rayleigh tuned IDs.")
    parser.add_argument("--save-root", default="/ceph/branco/Jasmine_Laurence/rayleigh_analysis/Top2_TunED")
    parser.add_argument("--experimental-root", default="/ceph/branco/Jasmine_Laurence/Experimental_Data")
    parser.add_argument("--nbins", type=int, default=30)
    parser.add_argument("--fps", type=float, default=40.0)
    parser.add_argument("--session-prefix", action="append", default=None, help="Optional prefix filter. Repeatable.")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-escapes-per-cell", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    save_root = Path(args.save_root)
    experimental_root = Path(args.experimental_root)

    a_csv = save_root / "tuned_ids_A_preflip_AND_maxrayleighA.csv"
    b_csv = save_root / "tuned_ids_B_postflip_AND_maxrayleighB.csv"
    out_dir = save_root / "escape_heatmaps_precomputed"
    npz_dir = out_dir / "npz"
    index_csv = out_dir / "escape_heatmaps_index.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(a_csv, b_csv)
    if args.session_prefix:
        pref = tuple(str(x).lower() for x in args.session_prefix)
        targets = targets[targets["session"].astype(str).str.lower().str.startswith(pref)].copy()
    if args.max_cells is not None:
        targets = targets.head(int(args.max_cells)).copy()

    session_to_processed = find_session_processed_dirs(experimental_root)
    resolved_map, unresolved, ambiguous = resolve_session_names(
        targets["session"].tolist(), list(session_to_processed.keys())
    )
    targets["session_discovered"] = targets["session"].map(resolved_map)
    keep = targets.dropna(subset=["session_discovered"]).copy()

    print(f"Targets after filters: {len(targets):,}")
    print(f"Resolved rows: {len(keep):,}")
    print(f"Resolved sessions: {keep['session'].nunique():,}")
    print(f"Unresolved sessions: {len(unresolved):,}")
    print(f"Ambiguous sessions: {len(ambiguous):,}")

    index_rows = []
    saved_npz = 0
    skipped_sessions = 0
    skipped_cells = 0
    cells_no_escape = 0

    for session, grp in keep.groupby("session", sort=True):
        sess_key = str(grp["session_discovered"].iloc[0])
        processed_dir = session_to_processed.get(sess_key)
        if processed_dir is None:
            skipped_sessions += len(grp)
            continue

        needed_clusters = grp["cluster_id"].astype(int).unique().tolist()
        vdf_raw, sdf_pd = load_session_data(processed_dir, needed_clusters=needed_clusters)
        if vdf_raw is None or sdf_pd is None:
            skipped_sessions += len(grp)
            continue
        shelter_loc, barrier_loc = load_arena_locations(processed_dir)

        spike_clusters = set(pd.to_numeric(sdf_pd["spike_clusters"], errors="coerce").dropna().astype(int).unique())
        pos_stats = vdf_raw.select(
            pl.col("mouse_x_position").min().alias("x_min"),
            pl.col("mouse_x_position").max().alias("x_max"),
            pl.col("mouse_y_position").min().alias("y_min"),
            pl.col("mouse_y_position").max().alias("y_max"),
        ).to_dicts()[0]
        eps = 1e-9
        x_edges = np.linspace(float(pos_stats["x_min"]), float(pos_stats["x_max"]) + eps, args.nbins + 1)
        y_edges = np.linspace(float(pos_stats["y_min"]), float(pos_stats["y_max"]) + eps, args.nbins + 1)

        print(f"\nSession {session} ({sess_key}) | rows={len(grp)}")
        for _, row in grp.iterrows():
            cluster_id = int(row["cluster_id"])
            tuned_label = str(row["tuned_label"])
            condition = str(row["condition"])
            if cluster_id not in spike_clusters:
                skipped_cells += 1
                continue

            bouts = extract_escape_bouts(vdf_raw, condition)
            if args.max_escapes_per_cell is not None:
                bouts = bouts[: int(args.max_escapes_per_cell)]
            if len(bouts) == 0:
                cells_no_escape += 1
                continue

            clu_spk = sdf_pd[sdf_pd["spike_clusters"] == cluster_id][["frame", "spike_count"]].copy()
            if clu_spk.empty:
                skipped_cells += 1
                continue

            for i, bout_frames in enumerate(bouts, start=1):
                esc_df = (
                    vdf_raw.filter(pl.col("frames").is_in(bout_frames.tolist()))
                    .select(["frames", "mouse_x_position", "mouse_y_position"])
                    .rename({"frames": "frame", "mouse_x_position": "x", "mouse_y_position": "y"})
                    .with_columns(pl.col("frame").cast(pl.Int64))
                    .to_pandas()
                )
                if esc_df.empty:
                    continue

                apply_bins(esc_df, x_edges, y_edges)
                rate, total_spikes, n_frames = build_rate_for_escape(esc_df, clu_spk, args.fps, args.nbins)
                fname = (
                    f"{session}__unit_{cluster_id}__{tuned_label}__{condition}__escape{i:02d}.npz"
                ).replace("/", "_")
                npz_path = npz_dir / fname
                np.savez_compressed(
                    npz_path,
                    rate=rate.astype(np.float32),
                    x_edges=x_edges.astype(np.float32),
                    y_edges=y_edges.astype(np.float32),
                )
                index_rows.append(
                    {
                        "session": session,
                        "session_discovered": sess_key,
                        "cluster_id": cluster_id,
                        "tuned_label": tuned_label,
                        "condition": condition,
                        "escape_idx": i,
                        "total_spikes": total_spikes,
                        "n_frames": n_frames,
                        "shelter_location_json": json.dumps(shelter_loc) if shelter_loc is not None else "",
                        "barrier_location_json": json.dumps(barrier_loc) if barrier_loc is not None else "",
                        "npz_path": str(npz_path),
                    }
                )
                saved_npz += 1

                del esc_df, rate
                gc.collect()

            del clu_spk
            gc.collect()

        del vdf_raw, sdf_pd
        gc.collect()

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(index_csv, index=False)

    print("\nDone precomputing escape heatmaps.")
    print(f"Saved npz heatmaps: {saved_npz}")
    print(f"Index CSV: {index_csv}")
    print(f"Skipped rows (session/csv issues): {skipped_sessions}")
    print(f"Skipped cells (cluster/spike issues): {skipped_cells}")
    print(f"Cells with no escapes: {cells_no_escape}")


if __name__ == "__main__":
    main()
