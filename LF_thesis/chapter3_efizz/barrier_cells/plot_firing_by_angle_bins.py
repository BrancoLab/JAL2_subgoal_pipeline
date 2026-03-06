import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_ROOT = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/Top2_TunED")
INPUT_CSV = SAVE_ROOT / "linked_tuned_ids_to_rayleigh_long.csv"
OUTPUT_DIR = SAVE_ROOT / "angle_bin_histograms"
FALLBACK_OUTPUT_DIR = Path(__file__).resolve().parent / "angle_bin_histograms"

# Use the same bin construction as Rayleigh polar plotting code.
NUMBER_OF_BINS = 13  # bin edges -> 12 angle bins
START_RAD = -np.pi
STOP_RAD = np.pi


def generate_bins(number_of_bins, start=-np.pi, stop=np.pi):
    """Replicates behave_analysis.analyze.filtering_data.filtering_functions.generate_bins."""
    bin_angles = np.linspace(start, stop, number_of_bins)
    bin_angle_center = bin_angles[:-1] + (np.mean(np.diff(bin_angles)) / 2)
    return bin_angles, bin_angle_center


def parse_array(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    if pd.isna(val):
        return np.asarray([], dtype=float)
    s = str(val).strip()
    if not s:
        return np.asarray([], dtype=float)

    try:
        parsed = ast.literal_eval(s)
        arr = np.asarray(parsed, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        s = s.strip("[]")
        arr = np.fromstring(s, sep=" ", dtype=float)
        return arr[np.isfinite(arr)]


def build_long_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    bin_edges, bin_centers = generate_bins(number_of_bins=NUMBER_OF_BINS, start=START_RAD, stop=STOP_RAD)
    n_bins = len(bin_centers)

    records = []
    for _, row in df.iterrows():
        fr = parse_array(row.get("firing_rate_hz", np.nan))
        if fr.size == 0:
            continue

        m = min(fr.size, n_bins)
        for i in range(m):
            records.append(
                {
                    "session": row.get("session"),
                    "cluster_id": row.get("cluster_id"),
                    "condition": row.get("condition"),
                    "tuned_label": row.get("tuned_label"),
                    "angle": row.get("angle"),
                    "rayleigh": row.get("rayleigh"),
                    "bin_index": i,
                    "bin_center_rad": float(bin_centers[i]),
                    "bin_center_deg": float(np.degrees(bin_centers[i])),
                    "firing_rate_hz_bin": float(fr[i]),
                }
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    out["bin_label"] = out["bin_center_deg"].map(lambda x: f"{x:.0f}°")
    return out


def _hist_range(x: np.ndarray):
    if x.size == 0:
        return (0.0, 1.0)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return (0.0, max(1.0, hi + 1.0))
    return (lo, hi)


def plot_hist_all_bins(long_df: pd.DataFrame, out_dir: Path):
    bins_sorted = sorted(long_df["bin_index"].unique())
    n = len(bins_sorted)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle("Firing rate distributions by real-world angle bin (all cells)", fontsize=14)

    global_vals = long_df["firing_rate_hz_bin"].to_numpy(dtype=float)
    hrange = _hist_range(global_vals)

    for idx, b in enumerate(bins_sorted):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = long_df[long_df["bin_index"] == b]
        vals = sub["firing_rate_hz_bin"].to_numpy(dtype=float)

        ax.hist(vals, bins=30, range=hrange, color="#2a9d8f", alpha=0.85, edgecolor="white")
        if vals.size:
            ax.axvline(
                float(np.nanmean(vals)),
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Mean",
            )
        deg = float(sub["bin_center_deg"].iloc[0]) if not sub.empty else np.nan
        ax.set_title(f"Bin {b} ({deg:.0f}°)")
        ax.set_xlabel("Firing rate (Hz)")
        ax.set_ylabel("Count")

    # Hide unused panels.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = out_dir / "hist_firing_rate_by_angle_bin_all_cells.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_hist_by_tuned_label(long_df: pd.DataFrame, out_dir: Path):
    bins_sorted = sorted(long_df["bin_index"].unique())
    n = len(bins_sorted)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle("Firing rate distributions by angle bin (A_only vs B_only)", fontsize=14)

    global_vals = long_df["firing_rate_hz_bin"].to_numpy(dtype=float)
    hrange = _hist_range(global_vals)

    for idx, b in enumerate(bins_sorted):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = long_df[long_df["bin_index"] == b]

        vals_a = sub.loc[sub["tuned_label"] == "A_only", "firing_rate_hz_bin"].to_numpy(dtype=float)
        vals_b = sub.loc[sub["tuned_label"] == "B_only", "firing_rate_hz_bin"].to_numpy(dtype=float)

        if vals_a.size:
            ax.hist(vals_a, bins=30, range=hrange, color="#1f77b4", alpha=0.5, edgecolor="white", label="A_only")
            ax.axvline(
                float(np.nanmean(vals_a)),
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label="A_only mean",
            )
        if vals_b.size:
            ax.hist(vals_b, bins=30, range=hrange, color="#d62728", alpha=0.5, edgecolor="white", label="B_only")
            ax.axvline(
                float(np.nanmean(vals_b)),
                color="#d62728",
                linestyle="--",
                linewidth=1.5,
                label="B_only mean",
            )

        deg = float(sub["bin_center_deg"].iloc[0]) if not sub.empty else np.nan
        ax.set_title(f"Bin {b} ({deg:.0f}°)")
        ax.set_xlabel("Firing rate (Hz)")
        ax.set_ylabel("Count")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = out_dir / "hist_firing_rate_by_angle_bin_A_vs_B.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    out_dir = OUTPUT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = FALLBACK_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Warning: no write permission to {OUTPUT_DIR}. Using {out_dir} instead.")

    df = pd.read_csv(INPUT_CSV)
    long_df = build_long_by_bin(df)

    if long_df.empty:
        raise RuntimeError("No valid firing-rate arrays were found to plot.")

    long_csv = out_dir / "linked_tuned_ids_to_rayleigh_angle_bins_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"Saved: {long_csv}")

    print(f"Input rows: {len(df)}")
    print(f"Expanded rows (cell x angle_bin): {len(long_df)}")
    ordered_bins = (
        long_df[["bin_index", "bin_label"]]
        .drop_duplicates()
        .sort_values("bin_index")
    )
    print(f"Angle bins (ordered): {ordered_bins['bin_label'].tolist()}")

    plot_hist_all_bins(long_df, out_dir)
    plot_hist_by_tuned_label(long_df, out_dir)


if __name__ == "__main__":
    main()
