import os
import numba
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from behave_analysis.analyze.filtering_data.filtering_functions import (
    filter_video_dataframe,
    generate_bins,
)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.arena_plotting import Arena


def rayleigh_map(spike_data, video_data, clusters, session, conditions, cluster_Ids, settings, tracking_data):
    """This function looks at the firing in each spatial position bin and then computes the rayleigh amplitude and angle for the avg. firing at each hdir. It then plots that as an arrow on the arena."""

    # fixed variables
    number_of_bins = 9  # for binning hdir
    num_pos_bins = 10  # number of bins for mouse position

    # saving path - where to save the figures
    map_path = make_directory(
        os.path.join(
            session.base_path,
            session.processed_path,
            "spatial_firing",
            "rayleigh_map",
            settings.cluster_type,
        )
    )

    if not isinstance(conditions, list):
        conditions = [conditions]

    # for each cluster make a map for each condition
    for count_clu, clu in enumerate(cluster_Ids):
        # for clu in np.arange(np.shape(spike_data)[1]):
        start_t = time.time()

        # identify cluster
        this_cluster = clusters.filter(clusters["spike_clusters"] == [clu])
        category = this_cluster["cluster_group"].to_numpy()

        # figure set-up
        nrows, ncols = single_cluster_plot_setup(conditions)

        # loop through conditions to make maps  and filter video_df
        for count, c in enumerate(conditions):
            # define plotting axes
            counter = ((ncols) * (count + 1)) + 2
            ax = plt.subplot(nrows, ncols, counter)

            # filter video_df
            video_df = filter_video_dataframe(video_data, condition=c, outofshelter=True, exclude_escape=True)
            video_df = video_df.select(["frames", "hdir", "mouse_x_position", "mouse_y_position"])

            # extract and bin hdir
            bin_angles, bin_angle_center = generate_bins(number_of_bins=number_of_bins, start = -np.pi, stop = np.pi)
            hdir = video_df["hdir"].to_numpy()
            hdir = np.digitize(hdir, bin_angles)
            hdir = bin_angle_center[hdir - 1]

            # extract and bin mouse position
            # assuming asquare image of the arena
            bin_pos, bin_pos_center = generate_bins(num_pos_bins, 1, session.video.height)  # assuming asquare image of the arena
            position = np.vstack(
                [
                    video_df["mouse_x_position"].to_numpy(),
                    video_df["mouse_y_position"].to_numpy(),
                ]
            ).T
            position = np.digitize(position, bin_pos)
            position = bin_pos_center[position - 1]

            # firing of this cluster, in this condition
            # -1 because frames are 1 indexed
            X = spike_data[video_df["frames"].to_numpy() - 1, count_clu]

            # identify unique position ad hdir combinations
            _, position_hdir_indices = np.unique(np.vstack((position.T, hdir.T)), axis=1, return_inverse=True)

            # find avg. firing at each hdir and postion
            unique_hdir_pos_comb = np.unique(position_hdir_indices)
            cum_fr = np.bincount(position_hdir_indices, weights=X)
            instances = np.bincount(position_hdir_indices)
            avg_firing = cum_fr[unique_hdir_pos_comb] / instances[unique_hdir_pos_comb]

            # unique positions
            _, unique_pos_vector = np.unique(position.T, axis=1, return_inverse=True)

            r, theta, this_pos = compute_rayleigh_by_pos(avg_firing, position_hdir_indices, hdir, position, number_of_bins, unique_pos_vector)
            empty_pos = np.isnan(r)
            full_pos = np.logical_not(empty_pos)

            # plot at each position that has a rayleigh an arrow/line with length and orientation given by rayleigh
            Arena(ax=ax, shelter_coordinates=tracking_data["shelter_loc"], condition=c, barrier_coordinates=session.barrier_location)
            # base_plotting(ax, tracking_data, condition = c, session = session)
            ax.scatter(this_pos[empty_pos, 0], this_pos[empty_pos, 1], 5, "k")
            ax.quiver(
                this_pos[full_pos, 0],
                this_pos[full_pos, 1],
                r[full_pos] * np.cos(theta[full_pos]),
                r[full_pos] * np.sin(theta[full_pos]),
                angles="xy",
                scale_units="xy",
                color="r",
                width=0.005,
            )

            # Set aspect ratio to be equal
            ax.set_aspect("equal")

        end_t = time.time()
        tot = end_t - start_t
        print("Time for cluster " + str(clu) + " is: " + str(tot))
        # when done with all conditions for this cluster, save and close the figure
        plt.tight_layout()
        plt.savefig(str(map_path) + "/" + category[0] + "_cluster" + str(clu) + "_hdir_rayleigh_map.png")
        if settings.show_plots:
            plt.show()
        plt.close()


@numba.jit(nopython=True)
def compute_rayleigh_by_pos(avg_firing, position_hdir_indices, hdir, position, number_of_bins, unique_pos_vector):
    # 1. iterate over positions
    # 2. check that all angles at this position have a minimum time, if not skip
    # 3. compute rayleigh

    unique_pos = np.unique(unique_pos_vector)
    # initialize variables
    r = np.zeros(shape=(len(unique_pos)))
    theta = np.zeros(shape=(len(unique_pos)))
    this_pos = np.zeros(shape=(len(unique_pos), 2))

    # iterate over unique positions
    for i, p in enumerate(unique_pos):
        here = np.where(unique_pos_vector == p)[0]
        pos0 = position_hdir_indices[here]

        # find xy coordinate and hdir of this position
        this_pos[i, :] = position[here[0], :]
        hdir_this_pos = hdir[here]
        unique_hdir_this_pos = np.unique(hdir_this_pos)

        # check that all angles at this position have a minimum time
        counts = np.zeros(shape=(len(unique_hdir_this_pos)))
        for hp_count, hp in enumerate(unique_hdir_this_pos):
            counts[hp_count] = np.sum(hdir_this_pos == hp)

        # make sure the mouse sampled all hdirs in this position
        if np.logical_and(len(np.unique(pos0)) == number_of_bins - 1, np.amin(counts) > 40):  # set an arbitrary lower bound of 1 second
            # get the hdir at this position and the firing at each hdir
            firing_this_pos = avg_firing[np.unique(pos0)]

            # compute rayleigh
            if not np.sum(firing_this_pos) == 0:
                x = np.sum(np.cos(unique_hdir_this_pos) * (firing_this_pos)) / np.sum(firing_this_pos)
                y = np.sum(np.sin(unique_hdir_this_pos) * (firing_this_pos)) / np.sum(firing_this_pos)
                theta[i] = np.arctan2(y, x)
                r[i] = np.sqrt(x**2 + y**2)
        else:
            r[i] = np.nan

    return r, theta, this_pos


def single_cluster_plot_setup(all_conditions):
    """Make a figure for each cluster with heatmaps in all conditions of interest"""

    # Add one index for the titles
    nrows = len(all_conditions) + 1
    ncols = 2

    # Plot settings
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        width_ratios=[1] + [3] * (ncols - 1),
        height_ratios=[1] + [3] * (nrows - 1),
        wspace=0,
        hspace=0.4,
    )
    # gridspec sets ratios such titles are narrower than plots
    fig = plt.figure(figsize=(ncols * 5, nrows * 5))  # width, height
    axs_fontsize = 23

    # Add subtitles for each condition in first column
    for c_counter, c in enumerate(all_conditions):
        ax = plt.subplot(gs[c_counter + 1, 0])
        ax.text(
            0,
            0.5,
            c,
            rotation="horizontal",
            va="center",
            ha="center",
            fontsize=axs_fontsize,
        )
        ax.set_axis_off()
    return nrows, ncols
