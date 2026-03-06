"""A function to calculate spatial efficiency as in Shamash et al. 2021"""

# OS Lib
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

# matplotlib.use("TkAgg")
from sklearn.metrics.pairwise import cosine_similarity

from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.color_funcs import get_color_based_on_speed
from behave_analysis.database.computer_ID import get_computer_specific_paths
from behave_analysis.utils.creating_directories import make_directory


def spatial_efficiency(onset_frames, stimulus_durations, session, settings, trial_conditions, tracking_data, trial_type, plotting=True, interp=100, save_dir=[]):
    """
    Plot escape trajectories as well as optimal path
    """
    ntrials = len(onset_frames)
    nrows = 4
    ncols = 5

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path="", return_ceph=True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, "summary_plots", trial_type + "_plots"))
    match = re.search(r"(\d{4}_\d{2}_\d{2})T", session.file_path)  # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    # Determine number of figures
    if ntrials > 20:
        number_of_figures = int(ntrials // 20 + (ntrials % 20 > 0))
    else:
        number_of_figures = 1

    spatial_efficiency_value = np.empty(len(onset_frames))
    trajectory_length = []
    # Plot up to 20 trials per figure
    trial_counter = 0
    for figure in range(number_of_figures):
        if plotting:
            fig = plt.figure(figsize=(20, 20))
            gs = gridspec.GridSpec(nrows, ncols, wspace=0, hspace=0)
        for row in range(nrows):
            for col in range(ncols):
                if trial_counter == ntrials:
                    break

                onset_frame = onset_frames[trial_counter]
                stimulus_duration = stimulus_durations[trial_counter]
                condition = trial_conditions[trial_counter]
                if np.isnan(onset_frame):
                    continue

                # set up axes with shelt and barrier locations

                if plotting:
                    ax = fig.add_subplot(gs[row, col])
                    Arena(ax=ax, shelter_coordinates=tracking_data["shelter_loc"], condition=condition, barrier_coordinates=session.barrier_location)
                    # base_plotting(ax,tracking_data,condition, session = session)
                else:
                    ax = []
                real_x, real_y, trajectory_length_single_trial = plot_escape_trajectories(
                    int(onset_frame), int(stimulus_duration * session.video.fps), tracking_data, settings, interp, ax
                )
                trajectory_length = np.append(trajectory_length, trajectory_length_single_trial)
                optimal_x, optimal_y = plot_optimal_trajectories(int(onset_frame), tracking_data, condition, interp, ax)

                # cosine similarity
                cs = []
                for x, y, ox, oy in zip(real_x, real_y, optimal_x, optimal_y):
                    cs = np.append(cs, cosine_similarity(np.array([x - 512, y]).reshape(1, -1), np.array([ox - 512, oy]).reshape(1, -1)))
                spatial_efficiency_value[trial_counter] = np.mean(cs)
                if plotting:
                    ax.set_title(f"spatial efficiency = {spatial_efficiency_value[trial_counter]:.3f}")
                trial_counter += 1

        if plotting:
            # save figure in session dir
            filename = str(save_dir) + "/" + trial_type + f"_SpatialEfficiency_{figure}.eps"
            plt.savefig(filename)
            # save in summary dir
            filename = session.mouse + "_" + date_str + "_" + trial_type + f"_SpatialEfficiency_{figure}.png"
            plt.savefig(overall_path + "/" + filename)
            if settings.show_plots:
                plt.show()
            plt.close()

    return spatial_efficiency_value, trajectory_length


def plot_escape_trajectories(onset_frames, stimulus_durations, tracking_data, settings, interp=100, ax=[]):
    """
    Plot escape trajectories.
    homings/escapes are cropped to when mouse enters the shelter
    for spatial efficiency calculation, the trajectories are interpolated to a uniform lengh given by interp
    RETURNS: x_loc and y_loc are vectors of x and y position during escape
    dist is cumulative path length of the trajectory until shelter is reached
    """
    # compute and plot each trajectory
    x_loc = tracking_data["head_loc"][onset_frames : onset_frames + stimulus_durations, 0]
    y_loc = tracking_data["head_loc"][onset_frames : onset_frames + stimulus_durations, 1]
    speed = tracking_data["avg_Velocity"][onset_frames : onset_frames + stimulus_durations]
    # crop the points after the mouse has entered the shelter
    in_shelt_y = y_loc > tracking_data["shelter_loc"][0][1]
    in_shelt_x = np.logical_and(x_loc > tracking_data["shelter_loc"][0][0], x_loc < tracking_data["shelter_loc"][1][0])
    in_shelt = np.where(np.logical_and(in_shelt_x, in_shelt_y))[0]
    if len(in_shelt) > 0:
        x_loc = x_loc[: in_shelt[0]]
        y_loc = y_loc[: in_shelt[0]]
        speed = speed[: in_shelt[0]]

    if len(x_loc) == 0:
        print("oppsie")
        return tracking_data["head_loc"][onset_frames, 0], tracking_data["head_loc"][onset_frames, 1], 0
    trail_color = np.empty((len(speed), 3))

    distance_travelled = []
    for i, stim_status in enumerate(np.arange(len(speed))):
        if ax:
            trail_color[i, :] = get_color_based_on_speed(speed=speed[i], object_to_color="trail", stim_status=stim_status, stim_type=settings.stim_type)
        if i > 0:
            distance_travelled = np.append(distance_travelled, np.sqrt((x_loc[i] - x_loc[i - 1]) ** 2 + (y_loc[i] - y_loc[i - 1]) ** 2))
    if ax:
        ax.scatter(x_loc, y_loc, s=5, c=trail_color / 255)

    # interpolate to standard size
    x_loc = np.interp(np.arange(0, len(x_loc), len(x_loc) / interp), np.arange(len(x_loc)), x_loc)
    y_loc = np.interp(np.arange(0, len(y_loc), len(y_loc) / interp), np.arange(len(y_loc)), y_loc)

    return x_loc, y_loc, np.sum(distance_travelled)


def plot_optimal_trajectories(onset_frames, tracking_data, condition, interp=100, ax=[]):
    """Plot optimal escape path"""
    opt_x = tracking_data["head_loc"][onset_frames, 0]
    opt_y = tracking_data["head_loc"][onset_frames, 1]
    opt_t = [0]
    # compute and plot each optimal trajectory to barrier
    if isinstance(condition, list):
        condition = condition[0]
    if not (any([condition == "shelter_only", condition == "pre_shelter", condition == "barrier_removed", opt_y > 512])):  # if no barrier or mouse starts in shelter zone
        opt_t = np.append(opt_t, (interp - 1) / 2)
        if condition == "barrier_present":  # double sided barrier
            nearest_barrier_edge = np.argmin(
                [
                    np.sqrt((opt_x - tracking_data["barrier_loc"][0][0]) ** 2 + (opt_y - tracking_data["barrier_loc"][0][1]) ** 2),
                    np.sqrt((opt_x - tracking_data["barrier_loc"][1][0]) ** 2 + (opt_y - tracking_data["barrier_loc"][1][1]) ** 2),
                ]
            )
        elif condition == "barrier_pre_flip":
            nearest_barrier_edge = 0
        elif condition == "barrier_post_flip":
            nearest_barrier_edge = 1

        opt_x = np.append(opt_x, tracking_data["barrier_loc"][nearest_barrier_edge][0])
        opt_y = np.append(opt_y, tracking_data["barrier_loc"][nearest_barrier_edge][1])

    opt_x = np.append(opt_x, np.mean([tracking_data["shelter_loc"][0][0], tracking_data["shelter_loc"][1][0]]))
    opt_y = np.append(opt_y, tracking_data["shelter_loc"][0][1])
    opt_t = np.append(opt_t, interp - 1)

    # interpolate optimal
    t_int = np.arange(interp)
    opt_xn = np.interp(t_int, opt_t, opt_x)
    opt_yn = np.interp(t_int, opt_t, opt_y)

    # compute and plot each optimal trajectory to shelter
    if ax:
        ax.scatter(opt_xn, opt_yn, s=5, c="r")

    return opt_xn, opt_yn
