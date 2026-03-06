from pathlib import Path
import sys

import pandas as pd


SAVE_ROOT = Path(r"/ceph/branco/Jasmine_Laurence/rayleigh_analysis/Top2_TunED")
CSV_PATH = SAVE_ROOT / "linked_tuned_ids_to_rayleigh_long.csv"
PKL_PATH = SAVE_ROOT / "linked_tuned_ids_to_rayleigh.pkl"
KEY_COLS = ["session", "cluster_id", "condition", "tuned_label", "angle"]


def _safe_stats(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "n/a"
    return f"min={vals.min():.4g}, median={vals.median():.4g}, max={vals.max():.4g}"


def summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")

    if "tuned_label" in df.columns:
        print("Tuned label counts:")
        print(df["tuned_label"].value_counts(dropna=False).to_string())

    if "rayleigh" in df.columns:
        ray = pd.to_numeric(df["rayleigh"], errors="coerce")
        print(f"Rayleigh non-null rows: {int(ray.notna().sum())}")
        print(f"Rayleigh stats: {_safe_stats(ray)}")

    if "firing_rate_mean_hz" in df.columns:
        print(f"Firing rate mean stats: {_safe_stats(df['firing_rate_mean_hz'])}")

    if all(c in df.columns for c in KEY_COLS):
        dupes = int(df.duplicated(KEY_COLS).sum())
        print(f"Duplicate key rows ({', '.join(KEY_COLS)}): {dupes}")

    print("\nPreview (first 10 rows):")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.head(10).to_string(index=False))


def compare_tables(csv_df: pd.DataFrame, pkl_df: pd.DataFrame):
    print("\n=== CSV vs PKL check ===")
    if list(csv_df.columns) != list(pkl_df.columns):
        print("Column mismatch between CSV and PKL")
        csv_only = [c for c in csv_df.columns if c not in pkl_df.columns]
        pkl_only = [c for c in pkl_df.columns if c not in csv_df.columns]
        print(f"CSV-only columns: {csv_only}")
        print(f"PKL-only columns: {pkl_only}")
        return

    if len(csv_df) != len(pkl_df):
        print(f"Row mismatch: CSV={len(csv_df)} PKL={len(pkl_df)}")
        return

    # Normalize index and compare exact values where possible.
    equal = csv_df.reset_index(drop=True).equals(pkl_df.reset_index(drop=True))
    if equal:
        print("CSV and PKL match exactly (row order + values).")
    else:
        print("CSV and PKL differ in at least one value or dtype.")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"PKL not found: {PKL_PATH}")

    csv_df = pd.read_csv(CSV_PATH)
    pkl_df = None

    # Backward/forward numpy pickle compatibility shim.
    try:
        import numpy.core.numeric as np_core_numeric

        sys.modules.setdefault("numpy._core.numeric", np_core_numeric)
    except Exception:
        pass

    try:
        pkl_df = pd.read_pickle(PKL_PATH)
    except Exception as exc:
        print("\nWarning: could not load PKL file.")
        print(f"Reason: {exc}")
        print("Proceeding with CSV-only summary.")

    summarize(csv_df, "CSV")
    if pkl_df is not None:
        summarize(pkl_df, "PKL")
        compare_tables(csv_df, pkl_df)


if __name__ == "__main__":
    main()
