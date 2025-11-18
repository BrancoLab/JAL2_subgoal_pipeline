import os
import re
from collections import defaultdict, Counter
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import polars as pl
import loguru as logger
import matplotlib

from settings.settings_analyze_efizz import Settings_ae
from behave_analysis.process.session import get_experiment
from behave_analysis.utils.rayleigh.load_rayleigh import collect_all_rayleigh_paths, load_all_rayleigh_data
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values, extract_firing_rates
from behave_analysis.utils.creating_directories import make_directory
from settings.settings_analyze_efizz import Settings_ae as Settings
from behave_analysis.analyze.TunED.model import TunEdModel

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr, JAL7_30apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may

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

tinny_barrier = [JAL8_tiny_3may, JAL8_21may, JAL7_30apr]

# Mice groups based on session names
mice_groups = {
    "JAL6": ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may"],
    "JAL4": ["JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"],
    "JAL5": ["JAL5_8thSept", "JAL5_21stSept"],
}


session_NAMES = [
    "JAL6_flip7_1apr",
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL005_8thSept",
    "JAL005_21stSept",
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

conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
dir = make_directory(r"Z:\Jasmine_Laurence\rayleigh_analysis")
angle_keys = ["hdir_Rayleigh.arrow", "hsa_Rayleigh.arrow", "h_postflipbar_a_Rayleigh.arrow", "h_preflipbar_a_Rayleigh.arrow"]


def regex(angle):
    "Remove unwanted characters from angle file string"
    pattern = r"^(.*?)(?=_Rayleigh)"
    match = re.search(pattern, angle)
    assert match, f"Could not find match for {angle}"
    return match.group(0)


def nest_dic():
    """A function to create arbitrarily nested dictionaries"""
    return defaultdict(nest_dic)


def build_threat_dict_max_rayleigh(
    experiments_objects,
    session_names,
    conditions,
    angle_keys,
):
    """
    Recreates your per-session sweep loop as a function (no global state).
    Returns:
        threat_dict_max_rayleigh : dict[session_name][cluster_id][condition] -> [(angle,label_rayleigh), (angle,label_rayleigh)] | "Not tuned"
        stats : dict with {'not_meet_threshold': int, 'cell_count': int, 'skipped_sessions': list}
    Assumes the following helper functions/objects exist in scope:
        - get_experiment(session)
        - collect_all_rayleigh_paths(session, cluster_type="good", conditions=...)
        - load_all_rayleigh_data(paths)
        - extract_firing_rates(condition_data[cond][angle])
        - extract_compartment_values(condition_data[cond][angle], column_name="Rayleigh")
        - regex(angle)
        - Each loaded_session has attributes: base_path, processed_path
    """

    dict = nest_dic()
    cell_count = 0
    skipped_sessions = []

    for i, session in enumerate(experiments_objects):
        sess_name = session_names[i]
        print(f"Processing session {i+1}/{len(experiments_objects)}: {session}")
        loaded_session = get_experiment(session)
        paths = collect_all_rayleigh_paths(session=loaded_session, cluster_type="good", conditions=conditions)  # paths[condition][angles]
        condition_data = load_all_rayleigh_data(paths)  # condition_data[condition][angles]

        # Load cluster master
        try:
            cluster_master = pd.read_csv(os.path.join(loaded_session.base_path, loaded_session.processed_path, "spike_count_by_frame_and_goodcluster.csv"))
        except Exception:
            print(f"Could not load cluster master for session {session}")
            skipped_sessions.append(session)
            continue

        spike_clusters = sorted(cluster_master["spike_clusters"].unique())

        # If no data for shelter_only, skip
        if len(condition_data["shelter_only"].keys()) == 0:
            print(f"Skipping session {session} as no data found")
            skipped_sessions.append(session)
            continue

        # Cells are same across conditions/angles in a session
        nCells = len(condition_data["shelter_only"]["hdir_Rayleigh.arrow"]["Rayleigh"])
        assert nCells == len(spike_clusters), f"Number of cells {nCells} does not match number of clusters {len(spike_clusters)}"
        cell_count += nCells

        for cell in range(nCells):
            for ci, condition in enumerate(condition_data.keys()):

                rayleigh_list_shelter = []
                rayleigh_list_threat = []
                firing_list_shelter = []
                firing_list_threat = []
                angle_list = []

                for angle in angle_keys:

                    angle_name = regex(angle)

                    # firing rate
                    sz_fr, tz_fr = extract_firing_rates(condition_data[condition][angle])
                    cell_fr_threat = tz_fr[cell]  # threat-zone firing rate only
                    cell_fr_shelter = sz_fr[cell]  # shelter-zone firing rate only

                    # Rayeliggh
                    r_output = extract_compartment_values(condition_data[condition][angle], column_name="Rayleigh")
                    rayleigh_threat = r_output[cell][1]  # 1 is threat 0 is shelter
                    rayleigh_shelter = r_output[cell][0]

                    # Store
                    firing_list_shelter.append(cell_fr_shelter)
                    firing_list_threat.append(cell_fr_threat)
                    rayleigh_list_shelter.append(rayleigh_shelter)
                    rayleigh_list_threat.append(rayleigh_threat)
                    angle_list.append(regex(angle))

                    dict[sess_name][spike_clusters[cell]][condition][angle_name] = {
                        "rayleigh_shelter": rayleigh_shelter,
                        "rayleigh_threat": rayleigh_threat,
                        "firing_rate_shelter": cell_fr_shelter,
                        "firing_rate_threat": cell_fr_threat,
                    }

    return dict


if __name__ == "__main__":
    re_run = False
    if re_run:

        results = build_threat_dict_max_rayleigh(
            experiments_objects=experiments_objects,
            session_names=session_NAMES,
            conditions=conditions,
            angle_keys=angle_keys,
        )

        # Save results
        output_path = os.path.join(dir, "threat_dict_max_rayleigh.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
    else:
        # Load results
        input_path = os.path.join(dir, "threat_dict_max_rayleigh.pkl") #BUG the name here is wrong it is not just threat zone its also compartment
        with open(input_path, "rb") as f:
            results = pickle.load(f)

        # plots
        # =========================
        # PLOTTING / EXPORT SECTION
        # =========================
        from pathlib import Path
        import math
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as backend_pdf
        import pandas as pd
        import numpy as np

        # 0) Output dirs
        plots_dir = os.path.join(dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        combined_pdf_path = os.path.join(plots_dir, "rayleigh_firing_plots.pdf")

        # 1) Flatten results -> tidy DataFrame
        def results_to_df(results_dict):
            rows = []
            for session_name, clusters in results_dict.items():
                for cluster_id, conds in clusters.items():
                    for condition, angles in conds.items():
                        for angle, metrics in angles.items():
                            # two compartments → two rows
                            rows.append(
                                {
                                    "session": session_name,
                                    "cluster_id": cluster_id,
                                    "condition": condition,
                                    "angle": angle,
                                    "compartment": "shelter",
                                    "firing_rate_hz": metrics["firing_rate_shelter"],
                                    "rayleigh": metrics["rayleigh_shelter"],
                                }
                            )
                            rows.append(
                                {
                                    "session": session_name,
                                    "cluster_id": cluster_id,
                                    "condition": condition,
                                    "angle": angle,
                                    "compartment": "threat",
                                    "firing_rate_hz": metrics["firing_rate_threat"],
                                    "rayleigh": metrics["rayleigh_threat"],
                                }
                            )
            df = pd.DataFrame(rows)
            # enforce categorical ordering for angles if helpful
            angle_order = ["hdir", "hsa", "h_postflipbar_a", "h_preflipbar_a"]

            def clean_angle(a):
                # ensure angles like "hdir_Rayleigh" are normalized
                # our dict keys already come from regex(angle), which returns e.g. 'hdir'
                return str(a).replace("_Rayleigh", "")

            df["angle"] = df["angle"].map(clean_angle)
            # keep order where present
            df["angle"] = pd.Categorical(df["angle"], categories=[a for a in angle_order if a in df["angle"].unique()], ordered=True)
            df["condition"] = pd.Categorical(
                df["condition"], categories=[c for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"] if c in df["condition"].unique()], ordered=True
            )
            df["compartment"] = pd.Categorical(df["compartment"], categories=["shelter", "threat"], ordered=True)
            return df

        # df = results_to_df(results)

        # # Save flattened data for downstream analysis
        # df_out = os.path.join(dir, "threat_dict_max_rayleigh_flat.csv")
        # df.to_csv(df_out, index=False)
        # print(f"Saved flattened data → {df_out}")

        # load the dataframe
        df = pd.read_csv(os.path.join(dir, "threat_dict_max_rayleigh_flat.csv"))
        print(df.columns)
        print(df.head())

        # Filter out rows where rayleigh value is < 0.25
        df = df[df["rayleigh"] >= 0.25].reset_index(drop=True)

        # ===== NEW PLOTTING PIPELINE (rewrite) =====
        import ast
        from matplotlib.backends import backend_pdf

        # Order categories
        angle_order = ["hdir", "hsa", "h_postflipbar_a", "h_preflipbar_a"]
        cond_order = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
        comp_order = ["shelter", "threat"]

        # Coerce categoricals (keep only those present)
        df["angle"] = pd.Categorical(df["angle"], categories=[a for a in angle_order if a in df["angle"].unique()], ordered=True)
        df["condition"] = pd.Categorical(df["condition"], categories=[c for c in cond_order if c in df["condition"].unique()], ordered=True)
        df["compartment"] = pd.Categorical(df["compartment"], categories=[c for c in comp_order if c in df["compartment"].unique()], ordered=True)

        # Parse firing_rate_hz column (it is an array serialized as a string in the CSV)
        def parse_array(val):
            if isinstance(val, (list, np.ndarray)):
                return np.asarray(val, dtype=float)
            if isinstance(val, (float, int)):
                return np.asarray([val], dtype=float)
            if isinstance(val, str):
                s = val.strip()
                # Try safe literal first, then fallback to fromstring
                try:
                    arr = ast.literal_eval(s)
                    return np.asarray(arr, dtype=float).ravel()
                except Exception:
                    s = s.strip("[]")
                    arr = np.fromstring(s, sep=" ")
                    return arr.astype(float) if arr.size else np.asarray([np.nan])
            return np.asarray([np.nan], dtype=float)

        # Per-row summaries of firing_rate_hz array
        df["fr_array"] = df["firing_rate_hz"].apply(parse_array)
        df["fr_mean"] = df["fr_array"].apply(lambda a: float(np.nanmean(a)) if a.size else np.nan)
        df["fr_median"] = df["fr_array"].apply(lambda a: float(np.nanmedian(a)) if a.size else np.nan)

        # Simple SEM helper
        def sem(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            n = len(x)
            return (np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan

        # Summary dataframe for plotting lines with error bars
        summary = (
            df.groupby(["angle", "condition", "compartment"], observed=True)
            .agg(
                mean_firing=("fr_mean", "mean"),
                sem_firing=("fr_mean", sem),
                mean_rayleigh=("rayleigh", "mean"),
                sem_rayleigh=("rayleigh", sem),
                n=("fr_mean", "size"),
            )
            .reset_index()
        )

        # Pretty labels
        nice_angle = {
            "hdir": "Head direction",
            "hsa": "Head–shelter angle",
            "h_postflipbar_a": "Post‑flip barrier angle",
            "h_preflipbar_a": "Pre‑flip barrier angle",
        }
        condition_labels = {
            "shelter_only": "Shelter only",
            "barrier_pre_flip": "Barrier (pre‑flip)",
            "barrier_post_flip": "Barrier (post‑flip)",
        }

        def label_angle(a):
            return nice_angle.get(str(a), str(a))

        def label_condition(c):
            return condition_labels.get(str(c), str(c))

        plots_dir = os.path.join(dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        combined_pdf_path = os.path.join(plots_dir, "rayleigh_firing_plots.pdf")

        def save_fig(fig, fname, pdf_writer=None):
            png_path = os.path.join(plots_dir, fname + ".png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            if pdf_writer is not None:
                pdf_writer.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {png_path}")

        # 1) Histograms per angle, split by compartment (rows) × condition (cols)
        def plot_histograms_by_angle_metric(df_in, metric_col, metric_label, bins=40, pdf_writer=None):
            angles = list(df_in["angle"].cat.categories) if hasattr(df_in["angle"], "cat") else sorted(df_in["angle"].unique())
            conds = list(df_in["condition"].cat.categories) if hasattr(df_in["condition"], "cat") else sorted(df_in["condition"].unique())
            compartments = list(df_in["compartment"].cat.categories) if hasattr(df_in["compartment"], "cat") else sorted(df_in["compartment"].unique())

            for angle in angles:
                sub_a = df_in[df_in["angle"] == angle]
                if sub_a.empty:
                    continue
                nrows, ncols = len(compartments), len(conds)
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
                fig.suptitle(f"{metric_label} — {label_angle(angle)}", fontsize=14)

                vmin = float(np.nanmin(sub_a[metric_col].to_numpy()))
                vmax = float(np.nanmax(sub_a[metric_col].to_numpy()))
                # Create a shared range so bins align visually
                rng = (vmin, vmax) if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax else None

                for i, comp in enumerate(compartments):
                    for j, cond in enumerate(conds):
                        ax = axes[i][j]
                        vals = sub_a[(sub_a["compartment"] == comp) & (sub_a["condition"] == cond)][metric_col].astype(float).to_numpy()
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        else:
                            ax.hist(vals, bins=bins, range=rng, color="#1f77b4", alpha=0.8, edgecolor="white")
                        ax.set_title(f"{label_condition(cond)} · {comp}")
                        ax.set_xlabel(metric_label)
                        ax.set_ylabel("Count")
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_fig(fig, f"hist_{metric_col}_{angle}", pdf_writer=pdf_writer)

        # 2) Mean ± SEM vs angle: one figure per compartment, lines = conditions
        def plot_mean_sem_vs_angle(summary_df, y_col_mean, y_col_sem, y_label, fname_prefix, pdf_writer=None):
            compartments = list(summary_df["compartment"].unique())
            angles = list(summary_df["angle"].cat.categories) if hasattr(summary_df["angle"], "cat") else sorted(summary_df["angle"].unique())
            conds = list(summary_df["condition"].cat.categories) if hasattr(summary_df["condition"], "cat") else sorted(summary_df["condition"].unique())
            x_pos = np.arange(len(angles))

            for comp in compartments:
                sub = summary_df[summary_df["compartment"] == comp]
                fig, ax = plt.subplots(figsize=(7.0, 4.6))

                for k, cond in enumerate(conds):
                    sc = sub[sub["condition"] == cond]
                    means, sems = [], []
                    for a in angles:
                        row = sc[sc["angle"] == a]
                        if row.empty:
                            means.append(np.nan)
                            sems.append(np.nan)
                        else:
                            means.append(float(row[y_col_mean].iloc[0]))
                            sems.append(float(row[y_col_sem].iloc[0]))
                    ax.errorbar(x_pos, means, yerr=sems, fmt="o-", capsize=3, lw=1.6, label=label_condition(cond))

                ax.set_xticks(x_pos)
                ax.set_xticklabels([label_angle(a) for a in angles], rotation=15, ha="right")
                ax.set_xlabel("Angle")
                ax.set_ylabel(y_label)
                ax.set_title(f"{y_label} vs angle — {comp} compartment")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(frameon=False)
                fig.tight_layout()
                save_fig(fig, f"{fname_prefix}_vs_angle_{comp}", pdf_writer=pdf_writer)

        # Write all figures into one PDF
        with backend_pdf.PdfPages(combined_pdf_path) as pdf_writer:
            # FR histograms use per-cell mean firing rate
            plot_histograms_by_angle_metric(df, metric_col="fr_mean", metric_label="Mean firing rate (Hz)", bins=40, pdf_writer=pdf_writer)
            plot_mean_sem_vs_angle(summary, y_col_mean="mean_firing", y_col_sem="sem_firing", y_label="Mean firing rate (Hz)", fname_prefix="firing_rate", pdf_writer=pdf_writer)

            # Rayleigh
            plot_histograms_by_angle_metric(df, metric_col="rayleigh", metric_label="Rayleigh value", bins=40, pdf_writer=pdf_writer)
            plot_mean_sem_vs_angle(summary, y_col_mean="mean_rayleigh", y_col_sem="sem_rayleigh", y_label="Rayleigh value", fname_prefix="rayleigh", pdf_writer=pdf_writer)

        print(f"Combined multi-page PDF saved to: {combined_pdf_path}")

        # ---- New: 4-panel overlay histograms (Shelter vs Threat) for pre/post-flip barrier angles ----
        # ...existing code...

        def plot_overlap_barrier_histograms(df_in, bins=40):
            """
            4-panel figure with overlaid histograms for shelter (blue) vs threat (purple)
            for the barrier angles. Each subplot shows vertical mean lines per compartment,
            a stats panel placed beneath the legend, and significance stars (Wilcoxon paired
            p if available else paired t-test).

            Panels:
            (0,0) Firing rate — Pre-flip barrier angle
            (0,1) Rayleigh    — Pre-flip barrier angle
            (1,0) Firing rate — Post-flip barrier angle
            (1,1) Rayleigh    — Post-flip barrier angle

            Saves PNG and one-page PDF into `plots_dir` defined above.
            """
            import matplotlib.backends.backend_pdf as backend_pdf
            from scipy.stats import ttest_rel, wilcoxon, ttest_ind, mannwhitneyu

            # Ensure firing metric exists (pipeline normally created fr_mean already)
            if "fr_mean" not in df_in.columns:
                df_in["fr_mean"] = pd.to_numeric(df_in["firing_rate_hz"], errors="coerce")

            # Helper: fetch values for a compartment (for the hist overlay, unpaired)
            def get_vals(angle_key, cond_key, comp_key, metric_col):
                sub = df_in[(df_in["angle"] == angle_key) & (df_in["condition"] == cond_key) & (df_in["compartment"] == comp_key)]
                vals = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy()
                vals = vals[np.isfinite(vals)]
                return vals

            # Helper: paired arrays using session×cluster_id index (shelter vs threat)
            def get_paired_vals(angle_key, cond_key, metric_col):
                sub = df_in[(df_in["angle"] == angle_key) & (df_in["condition"] == cond_key)][["session", "cluster_id", "compartment", metric_col]].copy()
                if sub.empty:
                    return np.array([]), np.array([])
                piv = (
                    sub.pivot_table(
                        index=["session", "cluster_id"],
                        columns="compartment",
                        values=metric_col,
                        aggfunc="mean",
                    )
                    .reindex(columns=["shelter", "threat"])
                    .dropna()
                )
                a = pd.to_numeric(piv["shelter"], errors="coerce").to_numpy()
                b = pd.to_numeric(piv["threat"], errors="coerce").to_numpy()
                mask = np.isfinite(a) & np.isfinite(b)
                return a[mask], b[mask]

            # Colors
            col_shelter = "#1f77b4"  # blue
            col_threat = "#9467bd"  # purple

            # Panels and data specs
            specs = [
                # (row, col, angle, condition, metric_col, title, x_label)
                (0, 0, "h_preflipbar_a", "barrier_pre_flip", "fr_mean", "Firing rate — Pre‑flip barrier angle", "Mean firing rate (Hz)"),
                (0, 1, "h_preflipbar_a", "barrier_pre_flip", "rayleigh", "Rayleigh — Pre‑flip barrier angle", "Rayleigh value"),
                (1, 0, "h_postflipbar_a", "barrier_post_flip", "fr_mean", "Firing rate — Post‑flip barrier angle", "Mean firing rate (Hz)"),
                (1, 1, "h_postflipbar_a", "barrier_post_flip", "rayleigh", "Rayleigh — Post‑flip barrier angle", "Rayleigh value"),
            ]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

            for r, c, angle_key, cond_key, metric_col, title, xlab in specs:
                ax = axes[r, c]
                sh = get_vals(angle_key, cond_key, "shelter", metric_col)
                th = get_vals(angle_key, cond_key, "threat", metric_col)

                # Empty case
                if sh.size == 0 and th.size == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    ax.set_xlabel(xlab)
                    ax.set_ylabel("Count")
                    continue

                # Shared range/bins so overlays align
                both = np.concatenate([sh, th]) if sh.size and th.size else (sh if sh.size else th)
                vmin, vmax = float(np.nanmin(both)), float(np.nanmax(both))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
                bin_edges = np.linspace(vmin, vmax, bins + 1)

                # Hist overlays
                if sh.size:
                    ax.hist(sh, bins=bin_edges, color=col_shelter, alpha=0.55, label="Shelter", edgecolor="white")
                if th.size:
                    ax.hist(th, bins=bin_edges, color=col_threat, alpha=0.55, label="Threat", edgecolor="white")

                # Mean vertical lines
                mean_sh = float(np.nanmean(sh)) if sh.size else np.nan
                mean_th = float(np.nanmean(th)) if th.size else np.nan
                if np.isfinite(mean_sh):
                    ax.axvline(mean_sh, color=col_shelter, linestyle="--", linewidth=2, alpha=0.95)
                if np.isfinite(mean_th):
                    ax.axvline(mean_th, color=col_threat, linestyle="--", linewidth=2, alpha=0.95)

                # Stats (prefer paired; fallback to unpaired if too few pairs)
                a, b = get_paired_vals(angle_key, cond_key, metric_col)
                n_pairs = len(a)
                p_t = np.nan
                p_w = np.nan
                if n_pairs >= 3:
                    try:
                        p_t = float(ttest_rel(a, b, alternative="two-sided").pvalue)
                    except Exception:
                        p_t = np.nan
                    try:
                        if np.allclose(a, b):
                            p_w = 1.0
                        else:
                            p_w = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
                    except Exception:
                        p_w = np.nan
                else:
                    # unpaired fallback
                    if sh.size >= 2 and th.size >= 2:
                        try:
                            p_t = float(ttest_ind(sh, th, equal_var=False).pvalue)  # Welch t
                        except Exception:
                            p_t = np.nan
                        try:
                            p_w = float(mannwhitneyu(sh, th, alternative="two-sided").pvalue)
                        except Exception:
                            p_w = np.nan

                # Legend first (top-right), then compute bbox to place stats below it
                leg = ax.legend(frameon=False, loc="upper right")
                ax.grid(axis="y", alpha=0.3)

                # Convert legend bbox to axes coords and place stats box just below
                def _axes_y_below_legend(ax, legend_obj, gap_ax=0.02, default_y=0.72):
                    try:
                        fig.canvas.draw()
                        renderer = fig.canvas.get_renderer()
                        leg_bbox_disp = legend_obj.get_window_extent(renderer=renderer)
                        _, y0_ax = ax.transAxes.inverted().transform((leg_bbox_disp.x0, leg_bbox_disp.y0))
                        return max(0.02, y0_ax - gap_ax)
                    except Exception:
                        return default_y

                def ptxt(p):  # short p formatting
                    return "n/a" if not np.isfinite(p) else f"{p:.2g}"

                text_y = _axes_y_below_legend(ax, leg, gap_ax=0.03, default_y=0.72)
                stats_txt = f"n_s={sh.size}, n_t={th.size}, pairs={n_pairs}\n" f"paired t p={ptxt(p_t)}\n" f"wilcoxon p={ptxt(p_w)}"
                ax.text(
                    0.98, text_y, stats_txt, transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none")
                )

                # Significance stars (prefer Wilcoxon p)
                p_for_star = p_w if np.isfinite(p_w) else p_t

                def stars(p):
                    if not np.isfinite(p):
                        return "ns"
                    if p < 1e-3:
                        return "***"
                    if p < 1e-2:
                        return "**"
                    if p < 5e-2:
                        return "*"
                    return "ns"

                # Place stars near top, centered between the two mean lines
                y_top = ax.get_ylim()[1]
                if np.isfinite(mean_sh) or np.isfinite(mean_th):
                    x_star = np.nanmean([mean_sh, mean_th]) if (np.isfinite(mean_sh) and np.isfinite(mean_th)) else (mean_sh if np.isfinite(mean_sh) else mean_th)
                    ax.text(x_star, y_top * 0.92, stars(p_for_star), ha="center", va="top", fontsize=14, fontweight="bold", color="black")

                ax.set_title(title)
                ax.set_xlabel(xlab)
                ax.set_ylabel("Count")

            # limit the x axis on the firing rates plots to 50 hz
            axes[0, 0].set_xlim(0, 30)
            axes[1, 0].set_xlim(0, 30)

            # Save PNG + a single-page PDF
            out_png = os.path.join(plots_dir, "overlap_histograms_barrier_pre_post.eps")
            fig.savefig(out_png, dpi=300, bbox_inches="tight", format="eps")
            with backend_pdf.PdfPages(os.path.join(plots_dir, "overlap_histograms_barrier_pre_post.pdf")) as pp:
                pp.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_png}")

        # ...existing code...

        # Create the 4-panel figure
        plot_overlap_barrier_histograms(df, bins=40)
        # ...existing code...
