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

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from scipy.ndimage import gaussian_filter

def single_unit_level_heatmaps(video_and_spike_data: pl.DataFrame, conditions, save_base) -> None:
    """"""
    save_path = make_directory(os.path.join(save_base, "single_unit_heatmaps"))
    unit_ids = video_and_spike_data["spike_clusters"].unique().to_numpy()
    fig, axs = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 15), sharey=True, sharex=True)
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],
    
    for clu in unit_ids:    
        for idx, condition in enumerate(conditions):
            filtered_df = filter_video_dataframe(dataframe=video_and_spike_data, condition=condition)
            clu_df = filtered_df.filter(pl.col("spike_clusters") == clu)
            
            if pddf.empty:  # Skip if no data for this unit and condition
                continue
            
            if sum(pddf["spike_count"]) == 0:  # Skip if no spikes for this unit and condition
                continue
            
            pddf = clu_df.to_pandas()
            
            # Bin the x and y positions
            x_bins = pd.cut(pddf["mouse_x_position"], bins=20, labels=False)  # Adjust bins as needed
            y_bins = pd.cut(pddf["mouse_y_position"], bins=20, labels=False)  # Adjust bins as needed
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
            pivot_df = normalized_df.pivot(index = "y_bins", columns = "x_bins", values = "normalized_spike_count").fillna(0)
            
            # Apply Gaussian smoothing
            if pivot_df.to_numpy().sum() > 0:  # Check if there's any data to smooth
                smoothed_data = gaussian_filter(pivot_df, sigma=0.2)
                sns.heatmap(smoothed_data, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs[idx], mask=(smoothed_data == 0))
                pivot_df = smoothed_data
            else:
                sns.heatmap(pivot_df, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs[idx], mask=(pivot_df == 0))
            
            # Create the heatmap
            axs[idx] = sns.heatmap(pivot_df, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs[idx],  mask=(pivot_df == 0))
            axs[idx].set_xticklabels([])
            axs[idx].set_yticklabels([])
            axs[idx].xaxis.set_ticks_position("none")
            axs[idx].yaxis.set_ticks_position("none")
            axs[idx].set_aspect("equal")
        
        fig.suptitle(f"Unit {clu}: heatmap", fontsize=20)
        file_name = f"unit{clu}_heatmap.png"
        fig.savefig(save_path + "\\" + file_name)
            
        # Exciplit memory management
        # del filtered_df, pddf, x_bins, y_bins, aggregated_df, bin_counts, normalized_df, pivot_df
        # gc.collect()
        plt.clf()
        plt.close('all')
    
    


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
        pivot_df = normalized_df.pivot(index = "y_bins", columns = "x_bins", values = "normalized_spike_count").fillna(0)
        
        # Apply Gaussian smoothing
        sigma = 0.2  # Standard deviation for Gaussian kernel. Adjust as needed.
        smoothed_data = gaussian_filter(pivot_df, sigma=sigma)
        
        # Create the heatmap
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), sharey=True, sharex=True)
        cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],
        axs = sns.heatmap(smoothed_data, cmap="coolwarm", robust=True, cbar_ax=cbar_ax, ax=axs,  mask=(pivot_df == 0))
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
        plt.close('all')
