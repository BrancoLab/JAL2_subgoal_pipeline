"""A module for creating heatmaps both at the population level and single level per condition to visualise 
how neurons are firing in relation to the mouse's position in the arena. No smoothing is applied to the data."""

import gc
import os

import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger

from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.utils.heatplot_utils import add_features, filter_outside_arena_tracking_for_video_and_spike_data

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


def single_unit_level_heatmaps(video_and_spike_data: pl.DataFrame, conditions: list, save_base: str, session: object) -> None:
    """Generate heatmaps for each unit split by condition and save each to a file.

    Logic:
        - Filter out data points outside of the arena
        - Bin the x and y positions
        - Return the total number of spikes in each x, y bin
        - Count the number of entries (data points) in each x, y bin
        - Merge the aggregated spike counts with the bin counts
        - Normalize the spike counts by the number of entries in each bin
        - Pivot the DataFrame for heatmap creation
        - Create the heatmap

    Args:
        video_and_spike_data (pl.DataFrame): A DataFrame containing spike counts per frame and neuron in a single session
        conditions (list): A list of conditions to divide the heatmap into, each condition is a string
        save_base (str): The base directory to save the heatmaps to
        session (Session): A Session object containing metadata about the session"""

    logger.info("Creating single unit heatmaps...")

    # Prepare the data
    tracking_data = open_tracking_data(session)
    save_path = make_directory(os.path.join(save_base, "single_unit_heatmaps"))
    data = video_and_spike_data.select(COLUMNS_TO_KEEP)  # Prevents memory issues and computer crashes
    del video_and_spike_data  # Release reference to the original DataFrame
    unit_ids = data["spike_clusters"].unique().to_numpy()
    data = filter_outside_arena_tracking_for_video_and_spike_data(video_and_spike_data=data, session=session)

    # Loop through each unit and create a heatmap for each condition
    for clu in unit_ids:
        total_spikes = 0
        fig, axs = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 7), sharey=True, sharex=True)
        cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])  # The list represents [left, bottom, width, height]
        for idx, condition in enumerate(conditions):
            filtered_df = filter_video_dataframe(dataframe=data, condition=condition)
            clu_df = filtered_df.filter(pl.col("spike_clusters") == clu)

            # Skip if no data for this unit and condition
            if clu_df.is_empty() or sum(clu_df["spike_count"]) == 0:
                continue

            pddf = clu_df.to_pandas()

            # Bin the x and y positions
            num_bins = 30  # Adjust bins as needed
            x_bins, x_bin_nums = pd.cut(pddf["mouse_x_position"], bins=num_bins, labels=False, retbins=True)
            y_bins, y_bin_nums = pd.cut(pddf["mouse_y_position"], bins=num_bins, labels=False, retbins=True)
            pddf["x_bins"] = x_bins
            pddf["y_bins"] = y_bins

            # Return the total number of spikes in each x, y bin
            spike_counts_in_bin = pddf.groupby(["x_bins", "y_bins"])["spike_count"].sum().reset_index()

            # Count the number of entries (data points) in each x, y bin
            total_entries_in_bins = pddf.groupby(["x_bins", "y_bins"]).size().reset_index(name="total_entries")

            # Merge the aggregated spike counts with the bin counts
            normalized_df = pd.merge(spike_counts_in_bin, total_entries_in_bins, on=["x_bins", "y_bins"])

            # Normalize the spike counts by the number of entries in each bin
            normalized_df["normalized_spike_count"] = normalized_df["spike_count"] / normalized_df["total_entries"]

            # Check that total entries is not longer than the inital dataframe
            assert sum(normalized_df["total_entries"]) <= len(pddf), "Total entries in bins is greater than the initial dataframe, can not be!"

            if condition != "all_time":
                total_spikes += sum(normalized_df["spike_count"])

            condition_spikes = sum(normalized_df["spike_count"])

            # Pivot the DataFrame for heatmap creation
            # Y bins are the rows, x bins are the columns and the values are firing rates
            pivot_df = normalized_df.pivot(index="y_bins", columns="x_bins", values="normalized_spike_count").fillna(0)

            # Create the heatmap
            add_features(axs[idx], condition, tracking_data, x_bin_nums, y_bin_nums)
            single_unit_heatmap_plotting(
                axs=axs, idx=idx, heatmap_data=pivot_df, cbar_ax=cbar_ax, condition=condition, condition_spikes=condition_spikes
            )

        fig.suptitle(f"Unit {clu}: heatmap - Total spikes across conditions {total_spikes:.1f}", fontsize=20)
        file_name = f"unit{clu}_heatmap.png"
        fig.savefig(save_path + "\\" + file_name)
        plt.clf()
        plt.close("all")

    logger.success("Single unit heatmaps created successfully!")


def single_unit_heatmap_plotting(axs, idx: int, heatmap_data: pl.DataFrame, cbar_ax, condition: str, condition_spikes) -> None:
    """Logic for plotting a single unit heatmap for a given condition

    Args:
        axs (matplotlib.axes): axis object to draw on
        idx (int): index of the current condition
        heatmap_data (pl.DataFrame): A DataFrame containing the heatmap data
        cbar_ax: The colorbar axis object
        condition (str): The condition of the plot"""
    axs[idx] = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        robust=True,
        cbar_ax=cbar_ax,
        ax=axs[idx],
        mask=(heatmap_data == 0),  # Mask zero values
    )
    axs[idx].set_xticklabels([])
    axs[idx].set_yticklabels([])
    axs[idx].set_title(condition + " \n spike #:" + str(int(condition_spikes)), fontsize=20)
    axs[idx].xaxis.set_ticks_position("none")
    axs[idx].yaxis.set_ticks_position("none")
    axs[idx].set_aspect("equal")
    cbar_ax.set_yticklabels([])


# -----------------------------------------------------------------------------------------------------------------------------
# - Don't seem too interesting for now but leaving code below for reference, might be useful in the future
# - Will primarily refactor and focus on the single unit level heatmaps
# - Will remove from the executing function so it won't be called


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
        del filtered_df, pddf, x_bins, y_bins, aggregated_df, bin_counts, normalized_df, pivot_df
        gc.collect()
        plt.clf()
        plt.close("all")
