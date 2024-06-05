"""
This is a script for making egocentric firing maps as in 
Alexander, A.S., Carstensen, L.C., Hinman, J.R., Raudies, F., Chapman, G.W., Hasselmo, M.E., 2020. Egocentric boundary vector tuning of the retrosplenial cortex. Sci Adv 6, eaaz2322. https://doi.org/10.1126/sciadv.aaz2322
"""

import os
import numba
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from behave_analysis.analyze.filtering_data.filtering_functions import (
    filter_video_dataframe,
    generate_bins,
)
from behave_analysis.utils.creating_directories import make_directory


def egocentric_firing_map(spike_data, video_data, clusters, session, conditions, cluster_Ids, settings):
    """This function sets up making a firing map for an egocentric view of features in the arena.
    For each cluster it will make a figure of egocentric firing maps in each condition.
    It will look at each position of the mouse, align the view of features in the arena based on the head angle of the mouse
    and then scale that view by the firing rate of the neuron of interest"""

    # fixed variables
    number_of_bins = 19  # for binning hdir
    num_pos_bins = 65  # number of bins for mouse position
    window_size = 200  # for cropping the rendered arena image around the mouse

    # saving path - where to save the figures
    map_path = make_directory(
        os.path.join(
            session.base_path,
            session.processed_path,
            "spatial_firing",
            "egocentric_map",
            settings.cluster_type,
        )
    )

    # make rendered arena image, add an offset for the cropping window
    # this defines the features that the firing map is built with
    x, y = generate_arena_feature_points(
        [session.video.height, session.video.width],
        session.shelter_location,
        session.barrier_location,
    )

    if isinstance(conditions, list):
        num_conditions = len(conditions)
    else:
        num_conditions = 1
        conditions = [conditions]

    # for each cluster make a map for each condition
    for count_clu, clu in enumerate(cluster_Ids):
        # for clu in np.arange(np.shape(spike_data)[1]):

        # identify cluster
        this_cluster = clusters.filter(clusters["spike_clusters"] == [clu])
        category = this_cluster["cluster_group"].to_numpy()

        # loop through conditions to make maps  and filter video_df
        heatmap = np.zeros(shape=(2 * window_size, 2 * window_size, num_conditions))
        for count, c in enumerate(conditions):
            video_df = filter_video_dataframe(video_data, condition=c, outofshelter=True, exclude_escape=True)
            video_df = video_df.select(["frames", "hdir", "mouse_x_position", "mouse_y_position"])

            # extract and bin hdir
            bin_angles, bin_angle_center = generate_bins(number_of_bins=number_of_bins, start = -np.pi, stop = np.pi)
            hdir = video_df["hdir"].to_numpy()
            hdir = np.digitize(hdir, bin_angles)
            hdir = bin_angle_center[hdir - 1]

            # extract and bin mouse position
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

            # heatmap for this cluster in this condition

            start_time = time.time()
            heatmap[:, :, count] = make_map_by_cluster_and_condition_speedy(X, position, hdir, [x, y], window_size)
            end_time = time.time()
            total_time = end_time - start_time
            print("For cluster ", clu, " and condition ", c, " total time:", total_time, "seconds ")

        single_cluster_plot(heatmap, conditions, settings, map_path, clu, category)


def make_map_by_cluster_and_condition_speedy(firing, position, hdir, arena, window_size):
    """
    This function will iterate over each position in the arena, and find the avg firing at each hdir.
    It then crops and rotates the arena image around the position and multiplies it by the avg firing rate.
    All of those cropped and scaled images are then summed to generate the map."""

    unique_pos, _ = np.unique(position, axis=0, return_inverse=True)
    unique_hdir = np.unique(hdir)

    # shift all arena points to center on the mouse's locations
    all_arena_points = np.tile(arena, len(unique_pos))
    all_arena_translations = np.repeat(unique_pos, len(arena[0]), axis=0)
    all_arena_points = all_arena_points.T - all_arena_translations  # (len(arena)*unique_pos) x 2

    # rotate by binned hdir
    all_rotation_matrices = np.array([[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in unique_hdir])
    # unique_hdir x 2 x (len(arena)*unique_pos)
    all_rotated_points = np.matmul(all_rotation_matrices, all_arena_points.T).astype(int)

    # this might be fewer than the optimal sampling which is len(unique_hdir)*len(unique_pos)
    position_hdir_combos, position_hdir_indices = np.unique(np.vstack((position.T, hdir.T)), axis=1, return_inverse=True)

    # find avg. firing at each hdir and postion
    unique_hdir_pos_comb = np.unique(position_hdir_indices)
    cum_fr = np.bincount(position_hdir_indices, weights=firing)
    instances = np.bincount(position_hdir_indices)
    avg_firing = cum_fr[unique_hdir_pos_comb] / instances[unique_hdir_pos_comb]

    # generate the map for each hdir + osition combination
    num_arena_points = len(arena[0])
    summed_rotated = generate_map(
        window_size, unique_pos, position_hdir_combos, num_arena_points, all_rotated_points, avg_firing, unique_hdir_pos_comb, unique_hdir
    )

    return summed_rotated


@numba.jit(nopython=True)
def generate_map(window_size, unique_pos, position_hdir_combos, num_arena_points, all_rotated_points, avg_firing, unique_hdir_pos_comb, unique_hdir):
    summed_rotated = np.zeros(shape=(2 * window_size, 2 * window_size))
    for count, c in enumerate(unique_hdir_pos_comb):
        # select arena at this position and angle bin
        axis0 = np.where(unique_hdir == position_hdir_combos[2, c])[0][0]
        axis2 = np.where(
            np.logical_and(
                unique_pos[:, 0] == position_hdir_combos[0, c],
                unique_pos[:, 1] == position_hdir_combos[1, c],
            )
        )[
            0
        ][0]
        this_pos_hdir = all_rotated_points[axis0, :, num_arena_points * axis2 : num_arena_points * (axis2 + 1)]

        # crop in window around mouse
        mask = np.logical_and(
            np.logical_and(this_pos_hdir[0, :] > -window_size, this_pos_hdir[0, :] < window_size),
            np.logical_and(this_pos_hdir[1, :] > -window_size, this_pos_hdir[1, :] < window_size),
        )
        this_pos_hdir = this_pos_hdir[:, mask] + window_size

        # scale & place positions on map
        rotated_image = np.zeros(shape=(2 * window_size, 2 * window_size))
        for x in np.unique(this_pos_hdir[1, :]):
            which_ones = this_pos_hdir[1, :] == x
            row = np.zeros(shape=(2 * window_size))
            row[this_pos_hdir[0, which_ones]] = np.ones(sum(which_ones)) * avg_firing[count]
            rotated_image[x, :] = row

        # this is more elegant but not allowed by numba
        # image_list = np.ravel_multi_index((this_pos_hdir[1,:],this_pos_hdir[0,:]),(2*window_size,2*window_size))
        # rotated_image = np.zeros(shape = ((2*window_size)*(2*window_size)))
        # rotated_image[image_list] = np.ones(len(image_list))*avg_firing[count]
        # rotated_image = np.reshape(rotated_image,(2*window_size,2*window_size))

        summed_rotated += rotated_image
    return summed_rotated


def single_cluster_plot(heatmap, all_conditions, settings, plot_save_path, cluster, category):
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

    # Plot the heatmaps
    for c_counter, c in enumerate(all_conditions):
        counter = ((ncols) * (c_counter + 1)) + 2
        ax = plt.subplot(nrows, ncols, counter)
        ax.imshow(heatmap[:, :, c_counter])
        ax.set_axis_off()

    # Save and close the figure
    plt.tight_layout()
    plt.savefig(str(plot_save_path) + "/" + category[0] + "_cluster" + str(cluster) + "_egocentric_map.png")
    if settings.show_plots:
        plt.show()
    plt.close()


# ---- UTILS


def generate_arena_feature_points(size, shelter_location, barrier_location):
    """This function generates a list of xy coordinates of all the relevant features (edges, barrier, shelter)
    that could cause the neurons to fire"""
    radius = 460
    num_points = 500
    theta = np.linspace(0, 2 * np.pi, num_points)
    # arena outline
    all_x = (radius * np.cos(theta)) + size[0] / 2
    all_y = radius * np.sin(theta) + size[1] / 2
    # shelter location
    x = np.tile(
        np.arange(shelter_location[0][0], shelter_location[1][0]),
        shelter_location[1][1] - shelter_location[0][1],
    )
    y = np.repeat(
        np.arange(shelter_location[0][1], shelter_location[1][1]),
        shelter_location[1][0] - shelter_location[0][0],
    )
    all_x = np.append(all_x, x)
    all_y = np.append(all_y, y)
    # barrier location, assumes horizontal barrier
    if barrier_location:
        x = np.arange(barrier_location[0][0], barrier_location[1][0])
        y = np.repeat(np.round(np.mean([barrier_location[0][1], barrier_location[1][1]])), len(x))
        all_x = np.append(all_x, x)
        all_y = np.append(all_y, y)
    return all_x, all_y


# ----------- OLD VERSION


def make_map_by_cluster_and_condition(firing, position, hdir, arena, window_size):
    """
    This old version is slower because so many for loops!
    ...
    This function will iterate over each position in the arena, and find the avg firing at each hdir.
    It then crops and rotates the arena image around the position and multiplies it by the avg firing rate.
    All of those cropped and scaled images are then summed to generate the map."""

    summed_rotated = np.zeros(shape=(2 * window_size, 2 * window_size))
    for x in np.unique(position[:, 0]):
        for y in np.unique(position[:, 1]):
            for h in np.unique(hdir):
                # avg neural activity in this bin
                avg_firing = 0
                this_bin = np.logical_and(np.logical_and(position[:, 0] == x, position[:, 1] == y), hdir == h)
                if sum(this_bin) > 0:  # only take average if the bin is not empty
                    avg_firing = np.nanmean(firing[this_bin])
                # crop the arena around this position
                crop = arena[
                    int(np.round(x - window_size)) : int(np.round(x + window_size)),
                    int(np.round(y - window_size)) : int(np.round(y + window_size)),
                ]
                image = Image.fromarray(crop.astype(np.uint8))
                # rotate the cropped image by hdir
                rotated_image = image.rotate(np.rad2deg(h))
                summed_rotated = summed_rotated + (np.array(rotated_image) * avg_firing)

    return summed_rotated


def generate_rendered_arena(size, shelter_location, barrier_location) -> object:
    """This function generates an image of the arena with all the relevant features (edges, barrier, shelter)
    that could cause the neurons to fire.
    Only used with the slower version of the mapping function"""
    rendered_arena = np.zeros(size).astype(np.uint8)
    # arena outline
    cv2.circle(
        rendered_arena,
        (int(size[0] / 2), int(size[1] / 2)),
        460,
        (255, 255, 255),
        thickness=2,
        lineType=16,
    )
    # shelter location
    cv2.rectangle(rendered_arena, (shelter_location[0]), (shelter_location[1]), 255, thickness=-1)
    # barrier location
    if barrier_location:
        cv2.line(
            rendered_arena,
            (barrier_location[0]),
            (barrier_location[1]),
            255,
            thickness=2,
        )
    return rendered_arena
