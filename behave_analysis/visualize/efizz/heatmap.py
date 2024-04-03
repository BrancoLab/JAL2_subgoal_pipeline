"""A module for creating heatmaps at the population level per condition to visualise arena level neural
activity across all neurons in a session.

Very slow and mem intensive."""

import gc
import os

import polars as pl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger
from tqdm import tqdm

from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from scipy.ndimage import gaussian_filter
from behave_analysis.utils.heatplot_utils import add_features

# Because of memory constraints, let's only keep the columns we need for this module
COLUMNS_TO_KEEP = [
    "mouse_x_position",
    "mouse_y_position",
    "spike_clusters",
    "spike_count",
    "OutofshelterIdx",
    "EscapePeriod",
    "shelter",
    "barrier_present",
    "barrier_flipped",
]


def single_unit_level_heatmaps(video_and_spike_data: pl.DataFrame, conditions, save_base, session) -> None:
    """"""

    logger.info("Creating single unit heatmaps...")
    tracking_data = open_tracking_data(session)
    save_path = make_directory(os.path.join(save_base, "single_unit_heatmaps"))
    data = video_and_spike_data.select(COLUMNS_TO_KEEP)  # Prevents memory issues and computer crashes
    del video_and_spike_data  # Release reference to the original DataFrame
    unit_ids = data["spike_clusters"].unique().to_numpy()

    # Remove all rows that are outside of the arena circle
    data = data.with_columns(
        np.sqrt(((pl.col("mouse_x_position") - session.video.height / 2) ** 2) + ((pl.col("mouse_y_position") - session.video.height / 2) ** 2)).alias("dist")
    )
    
    # Now filter out the mouse x positions where distance is less than 460 and same for y
    data = data.filter(pl.col("dist") < 460)
    data = data.drop("dist")
    
    for clu in unit_ids:
        fig, axs = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 10), sharey=True, sharex=True)
        cbar_ax = fig.add_axes([0.91, 0.6, 0.01, 0.2])  # The list represents [left, bottom, width, height]
        logger.info(f"Creating heatmap for unit {clu}")
        for idx, condition in enumerate(conditions):
            filtered_df = filter_video_dataframe(dataframe=data, condition=condition)
            clu_df = filtered_df.filter(pl.col("spike_clusters") == clu)

            if clu_df.is_empty():  # Skip if no data for this unit and condition
                continue

            if sum(clu_df["spike_count"]) == 0:  # Skip if no spikes for this unit and condition
                continue

            pddf = clu_df.to_pandas()

            # Bin the x and y positions
            x_bins, x_bin_nums = pd.cut(pddf["mouse_x_position"], bins=30, labels=False, retbins=True)  # Adjust bins as needed
            y_bins, y_bin_nums = pd.cut(pddf["mouse_y_position"], bins=30, labels=False, retbins=True)  # Adjust bins as needed

            pddf["x_bins"] = x_bins
            pddf["y_bins"] = y_bins

            # Return the total number of spikes in each x, y bin
            aggregated_df = pddf.groupby(["x_bins", "y_bins"])["spike_count"].sum().reset_index()

            # Count the number of entries (data points) in each x, y bin
            bin_counts = pddf.groupby(["x_bins", "y_bins"]).size().reset_index(name="total_entries")

            # Merge the aggregated spike counts with the bin counts
            normalized_df = pd.merge(aggregated_df, bin_counts, on=["x_bins", "y_bins"])

            # Normalize the spike counts by the number of entries in each bin - where 40 is frames per second
            normalized_df["normalized_spike_count"] = (normalized_df["spike_count"] / normalized_df["total_entries"]) * 40

            # Pivot the DataFrame for heatmap creation
            # Y bins are the rows, x bins are the columns and the values are firing rates
            pivot_df = normalized_df.pivot(index="y_bins", columns="x_bins", values="normalized_spike_count").fillna(0)

            # Create the heatmap
            add_features(axs[idx], condition, tracking_data, x_bin_nums, y_bin_nums)
            # axs[idx] = sns.heatmap(pivot_df, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs[idx], mask=(pivot_df == 0))
            axs[idx] = sns.heatmap(
                pivot_df,
                cmap="coolwarm",
                robust=True,
                cbar_ax=cbar_ax,
                ax=axs[idx],
                mask=(pivot_df == 0),  # Mask zero values
            )
            axs[idx].set_xticklabels([])
            axs[idx].set_yticklabels([])
            axs[idx].set_title(condition, fontsize=20)
            axs[idx].xaxis.set_ticks_position("none")
            axs[idx].yaxis.set_ticks_position("none")
            axs[idx].set_aspect("equal")
            cbar_ax.set_label("")  # Colour bar is last axis

        fig.suptitle(f"Unit {clu}: heatmap", fontsize=20)
        file_name = f"unit{clu}_heatmap.png"
        fig.savefig(save_path + "\\" + file_name)
        plt.clf()


def population_level_heatmaps(video_and_spike_data: pl.DataFrame, conditions, save_base) -> None:
    """Create a heatmap at the population level for all neurons in a session
    and save that heatmap to a file. Divide the heatmap into conditions and save
    as a separate file for each condition. Mainly due to memory constraints as was crashing
    when trying to plot all conditions at once.

    Args:
        spike_by_frame_df (pd.DataFrame): A dataframe containing spike counts per frame and neuron in a single session
        conditions (list): A list of conditions to divide the heatmap into, each condition is a string
        save_base (str): The base directory to save the heatmaps to
    """

    save_path = make_directory(os.path.join(save_base, "population_heatmaps"))

    for _, condition in enumerate(conditions):

        # Filter_video_dataframe is a function that filters the dataframe based on the given condition
        filtered_df = filter_video_dataframe(dataframe=video_and_spike_data, condition=condition)
        pddf = filtered_df.to_pandas()

        # Bin the x and y positions
        x_bins = pd.cut(pddf["mouse_x_position"], bins=25, labels=False)  # Adjust bins as needed
        y_bins = pd.cut(pddf["mouse_y_position"], bins=25, labels=False)  # Adjust bins as needed
        pddf["x_bins"] = x_bins
        pddf["y_bins"] = y_bins

        # Return the total number of spikes in each x, y bin
        aggregated_df = pddf.groupby(["x_bins", "y_bins"])["spike_count"].sum().reset_index()

        # Count the number of entries (data points) in each x, y bin
        bin_counts = pddf.groupby(["x_bins", "y_bins"]).size().reset_index(name="total_entries")

        # Merge the aggregated spike counts with the bin counts
        normalized_df = pd.merge(aggregated_df, bin_counts, on=["x_bins", "y_bins"])

        # Normalize the spike counts by the number of entries in each bin
        normalized_df["normalized_spike_count"] = normalized_df["spike_count"] / normalized_df["total_entries"] * 40

        # Pivot the DataFrame for heatmap creation
        # Y bins are the rows, x bins are the columns and the values are firing rates
        pivot_df = normalized_df.pivot(index="y_bins", columns="x_bins", values="normalized_spike_count").fillna(0)

        # Create the heatmap
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), sharey=True, sharex=True)
        cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],
        axs = sns.heatmap(pivot_df, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs, mask=(pivot_df == 0))
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.xaxis.set_ticks_position("none")
        axs.yaxis.set_ticks_position("none")
        axs.set_title(condition, fontsize=20)
        axs.set_aspect("equal")
        file_name = f"population_heatmap_{condition}.png"
        fig.savefig(save_path + "\\" + file_name)

        # Exciplit memory management
        del filtered_df, pddf, x_bins, y_bins, aggregated_df, bin_counts, normalized_df, pivot_df, smoothed_data
        gc.collect()
        plt.clf()
        plt.close("all")
