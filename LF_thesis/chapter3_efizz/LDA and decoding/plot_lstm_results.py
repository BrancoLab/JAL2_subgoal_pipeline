"""Plot LSTM decoding performance stored in LSTM_results.pkl."""

from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PKL = Path(r"Z:\Laurence\thesis\efizz_chapter\LSTM_results.pkl")
OUT_DIR = Path(r"Z:\Laurence\thesis\efizz_chapter") / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    if not RESULTS_PKL.exists():
        raise FileNotFoundError(f"{RESULTS_PKL} not found.")
    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)
    df = pd.DataFrame(results, columns=["session", "condition", "target", "test_r2"])
    return df


def plot_per_session(df: pd.DataFrame):
    sessions = df["session"].unique()
    for session in sessions:
        df_session = df[df["session"] == session]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(df_session))
        ax.bar(x, df_session["test_r2"], tick_label=df_session["condition"] + "\n" + df_session["target"], color="#74b9ff")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Test R²")
        ax.set_title(f"LSTM decoding {session}")
        ax.set_ylim(min(-0.1, df_session["test_r2"].min() - 0.05), df_session["test_r2"].max() + 0.05)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{session}_LSTM_decoding.png", dpi=200)
        plt.show()
        plt.close(fig)


def plot_overall_summary(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = df.groupby(["condition", "target"])["test_r2"].mean().reset_index()
    x = np.arange(len(grouped))
    ax.bar(x, grouped["test_r2"], color="#55efc4")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["condition"] + "\n" + grouped["target"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Test R²")
    ax.set_title("Average LSTM performance per condition/target")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "LSTM_overall_summary.png", dpi=200)
    plt.show()
    plt.close(fig)


def main():
    df = load_results()
    plot_per_session(df)
    plot_overall_summary(df)
    print("Saved plots for individual sessions and overall summary.")


if __name__ == "__main__":
    main()
