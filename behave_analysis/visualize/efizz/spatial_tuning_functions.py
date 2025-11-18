"""a set of functions for visualizing the tuning (e.g. spatial) of neurons"""

# set up
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import polars as pl

# import
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.visualize.behaviour.behavioral_stats import hsv_hdir_colormap


def spatial_position_firing_hdir(data, clu_label, video_df, save_path, show_plots):
    """
    A function that plots the position of the mouse at every AP of a given cluster and colours it by hdir
    """

    logger.info("Commence making figures of spatial position firing plots coloured by hdir of all clusters")
    cc = matplotlib.cm.Reds  # could use Reds or copper
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5  # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows, ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()

    video_df = filter_video_dataframe(video_df, "all_time")
    video_df = video_df.select(["frames", "hdir", "mouse_x_position", "mouse_y_position"])
    spike_data = data.select(["spike_clusters", "spike_aligned_to_frame"])

    # what is firing rate per frame?
    for counter, cluster in enumerate(data["spike_clusters"].unique()):
        if counter >= (ncols * nrows) * fnum:
            figg, axs = plt.subplots(nrows, ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()
        # filter spikes by cluster
        spikes = spike_data.filter(spike_data["spike_clusters"] == cluster)
        # align spike data for this cluster to video_df
        spikes = spikes.with_column(spikes["spike_aligned_to_frame"].cast(pl.Int64))
        merged_df = video_df.join(spikes, left_on="frames", right_on="spike_aligned_to_frame", how="inner")

        hdir = np.digitize(np.rad2deg(merged_df["hdir"]), np.arange(-180, 180))
        cc = hsv_hdir_colormap(hdir)

        axs[counter - (nrows * ncols * fnum)].scatter(
            video_df["mouse_x_position"].to_numpy(), video_df["mouse_y_position"].to_numpy(), s=3, color=[0.7, 0.7, 0.7], linewidths=0, marker="."
        )  # all mouse positions
        axs[counter - (nrows * ncols * fnum)].scatter(
            merged_df["mouse_x_position"].to_numpy(), merged_df["mouse_y_position"].to_numpy(), s=7, c=cc, linewidths=0, marker="."
        )  # this neuron's firing coloured by hdir
        axs[counter - (nrows * ncols * fnum)].set_axis_off()
        axs[counter - (nrows * ncols * fnum)].invert_yaxis()
        axs[counter - (nrows * ncols * fnum)].set_aspect("equal")
        this_cluster = clu_label.filter(clu_label["spike_clusters"] == [cluster])
        axs[counter - (nrows * ncols * fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + " cluster " + str(cluster))

        # save the figure
        if np.logical_or(counter - (nrows * ncols * (fnum - 1)) == (ncols * nrows) - 1, counter == len(data["spike_clusters"].unique()) - 1):

            plt.tight_layout()
            plt.savefig(str(save_path) + "_clusters_spatial_firing_hdir_colored" + str(fnum) + ".eps", format="eps")

            if show_plots:
                plt.show()

            plt.close()


def spatial_position_firing(data, clu_label, video_spike_count_df, save_path, show_plots):
    """
    A function that makes maps of mousie's position in arena and show where each cluster fired
    """
    # TODO: rewrite this using big postprocess matrix instead
    logger.info("Commence making figures of spatial position firing plots of all clusters")
    cc = matplotlib.cm.Reds  # could use Reds or copper
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5  # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows, ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()

    large_dataFrame = video_spike_count_df.select(["frames", "spike_clusters", "mouse_x_position", "mouse_y_position", "spike_count"])

    # what is firing rate per frame?
    for counter, cluster in enumerate(data["spike_clusters"].unique()):
        if counter >= (ncols * nrows) * fnum:
            figg, axs = plt.subplots(nrows, ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()
        # filter spikes by cluster
        spikes = large_dataFrame.filter(large_dataFrame["spike_clusters"] == cluster)
        spikes = spikes.fill_null(strategy="zero")

        axs[counter - (nrows * ncols * fnum)].scatter(
            spikes["mouse_x_position"].to_numpy(), spikes["mouse_y_position"].to_numpy(), s=5, c=cc(spikes["spike_count"].to_numpy() * 50), linewidths=0, marker="."
        )  # srate*2 increase contrast
        axs[counter - (nrows * ncols * fnum)].set_axis_off()
        axs[counter - (nrows * ncols * fnum)].invert_yaxis()
        axs[counter - (nrows * ncols * fnum)].set_aspect("equal")
        this_cluster = clu_label.filter(clu_label["spike_clusters"] == [cluster])
        axs[counter - (nrows * ncols * fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + " cluster " + str(cluster))

        # save the figure
        if np.logical_or(counter - (nrows * ncols * (fnum - 1)) == (ncols * nrows) - 1, counter == len(data["spike_clusters"].unique()) - 1):
            plt.tight_layout()
            plt.savefig(str(save_path) + "_clusters_spatial_position_firing_" + str(fnum) + ".png")

            if show_plots:
                plt.show()

            plt.close()


##----------- OLDER VERSIONS


def spatial_position_firing_old(self):
    """
    A function that makes maps of mousie's position in arena and show where each cluster fired
    """

    logger.info("Commence making figures of spatial position firing plots of all clusters")
    cc = matplotlib.cm.Reds  # could use Reds or copper
    # set number of rows and calculate number of columns
    ncols = 10
    nrows = 5  # nclu // ncols + (nclu % ncols > 0)
    figg, axs = plt.subplots(nrows, ncols)
    figg.set_figwidth(30)
    figg.set_figheight(15)
    fnum = 1
    axs = axs.ravel()

    large_dataFrame = self.processed_data.video_spike_count_df.select(["frames", "spike_clusters", "mouse_x_position", "mouse_y_position", "spike_count"])

    # what is firing rate per frame?
    for counter, cluster in enumerate(self.processed_data.spike_data["spike_clusters"].unique()):
        if counter >= (ncols * nrows) * fnum:
            figg, axs = plt.subplots(nrows, ncols)
            figg.set_figwidth(30)
            figg.set_figheight(15)
            fnum = fnum + 1
            axs = axs.ravel()
        # filter spikes by cluster
        spikes = large_dataFrame.filter(large_dataFrame["spike_clusters"] == cluster)
        spikes = spikes.fill_null(strategy="zero")

        axs[counter - (nrows * ncols * fnum)].scatter(
            spikes["mouse_x_position"].to_numpy(), spikes["mouse_y_position"].to_numpy(), s=5, c=cc(spikes["spike_count"].to_numpy() * 50), linewidths=0, marker="."
        )  # srate*2 increase contrast
        axs[counter - (nrows * ncols * fnum)].set_axis_off()
        axs[counter - (nrows * ncols * fnum)].invert_yaxis()
        axs[counter - (nrows * ncols * fnum)].set_aspect("equal")
        this_cluster = self.processed_data.clu_label.filter(self.processed_data.clu_label["spike_clusters"] == [cluster])
        axs[counter - (nrows * ncols * fnum)].title.set_text(str(this_cluster["cluster_group"].to_numpy()) + " cluster " + str(cluster))

        # save the figure
        if np.logical_or(counter - (nrows * ncols * (fnum - 1)) == (ncols * nrows) - 1, counter == len(self.processed_data.spike_data["spike_clusters"].unique()) - 1):
            plt.tight_layout()
            plt.savefig(str(self.spatial_path) + "/" + self.processed_data.select_clusters + "_clusters_spatial_position_firing_" + str(fnum) + ".png")

            if settings_v.show_plots:
                plt.show()

            plt.close()
