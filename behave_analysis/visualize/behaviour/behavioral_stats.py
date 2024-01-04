'''A set of functions for visualizing behavioral statistics of a mouse in a given session'''

# set up
import os
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, filter_video_dataframe

def position_by_bsa(tracking_data, outofShelterIdx, settings, save_path):
    """Make a scatter plot of position in arena colored by angle between body and shelter"""

    # color position by their shelter angle
    mass = tracking_data["avg_loc"][outofShelterIdx, :]
    ang_color = np.digitize(np.rad2deg(tracking_data["bod_shelt_dir"][outofShelterIdx]), np.arange(-180, 180))
    bsa_rgb_cycle = hsv_hdir_colormap(ang_color)
    plt.figure()
    plt.scatter(mass[:, 0], mass[:, 1], s=5, c=bsa_rgb_cycle, linewidths=0, marker=".")
    plt.title("position coloured by angle to shelter")
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.savefig(os.path.join(save_path, "arena_position.png"))
    if settings.show_plots:
        plt.show()
    plt.close()

def shelter_occupancy(video_df, session, settings, save_path):
    """Make a bar plot of minutes in and out of shelter per condition in each session"""

    if settings.user_defined_conditions:
        conditions = settings.conditions
    else:
        conditions = identify_conditions(session)

    _, ax = plt.subplots(figsize=(10, 5))

    for x, c in enumerate(conditions):
        # Plot out of shelter
        time_out_of_shelter = (
            len(filter_video_dataframe(video_df, c, outofshelter=True, exclude_escape=False))
            / session.video.fps
        )
        plt.bar(
            x + 0.9,
            time_out_of_shelter / 60,
            width=0.2,
            color="dimgrey",
        )

        # Plot in shelter
        time_in_shelter = (
            len(filter_video_dataframe(video_df, c, outofshelter=False, exclude_escape=False))
            / session.video.fps
        )
        plt.bar(
            x + 1.1,
            time_in_shelter / 60,
            width=0.2,
            color="lightgrey",
        )

    # Plot settings
    # sns.set()
    plt.xticks(np.arange(len(conditions)) + 1, conditions, rotation=35)
    plt.ylabel("Time (mins)")
    plt.legend(["out of shelter", "in shelter"], facecolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shelter_occupancy.png"))
    if settings.show_plots:
        plt.show()
    plt.close()

def location_occupancy(tracking_data, session, settings, save_path):
    """
    Make plots showing time in shelter, near barrier edged and in 4 quadrants of arena over the course of the session
    """
    # look in a 3 minute window
    w = 3
    # x axis values for plotting, in minutes
    x = np.arange(
        w / 2,
        (session.video.num_frames / (session.video.fps * 60))
        - (w / 2)
        + 1 / (session.video.fps * 60),
        1 / (session.video.fps * 60),
    )
    figg, axs = plt.subplots(1, 3)
    figg.set_figwidth(15)
    # time in shelter
    if len(session.shelter_time) > 0:
        if "mushroom" in session.experiment:
            extra = 50  # in the mushroom session extend what the shelter is beyond the base
        else:
            extra = 0
        InShelterIdx = np.logical_and(
            np.logical_and(
                tracking_data["avg_loc"][:, 0] > tracking_data["shelter_loc"][0][0] - extra,
                tracking_data["avg_loc"][:, 0] < tracking_data["shelter_loc"][1][0] + extra,
            ),
            np.logical_and(
                tracking_data["avg_loc"][:, 1] > tracking_data["shelter_loc"][0][1] - extra,
                tracking_data["avg_loc"][:, 1] < tracking_data["shelter_loc"][1][1] + extra,
            ),
        )
        axs[0].plot(
            x,
            np.convolve(InShelterIdx.astype(int), np.ones(session.video.fps * 60 * w), "valid")
            / (session.video.fps * 60 * w),
        )
        axs[0].plot(
            [session.shelter_time[0], session.shelter_time[0]], [0, 1], "-k"
        )
        axs[0].title.set_text("In shelter")
        axs[0].set_xlabel("time (mins)")
        axs[0].set_ylabel("fraction occupancy")

    # time in 4 quadrants
    cc = matplotlib.cm.Set1
    center = [session.video.width / 2, session.video.height / 2]
    Q = np.vstack(
        (
            np.logical_and(
                tracking_data["avg_loc"][:, 0] < center[0], tracking_data["avg_loc"][:, 1] < center[1]
            ),  # upper_left
            np.logical_and(
                tracking_data["avg_loc"][:, 0] > center[0], tracking_data["avg_loc"][:, 1] < center[1]
            ),  # upper_right
            np.logical_and(
                tracking_data["avg_loc"][:, 0] > center[0], tracking_data["avg_loc"][:, 1] > center[1]
            ),  # lower_right
            np.logical_and(
                tracking_data["avg_loc"][:, 0] < center[0], tracking_data["avg_loc"][:, 1] > center[1]
            ),
        )
    )  # lower_left
    for i in np.arange(4):
        axs[1].plot(
            x,
            np.convolve(Q[i, :].astype(int), np.ones(session.video.fps * 60 * w), "valid")
            / (session.video.fps * 60 * w),
            color=cc(i),
        )
    axs[1].title.set_text("In quadrants")
    axs[1].legend(["upper_left", "upper_right", "lower_right", "lower_left"])
    axs[1].set_xlabel("time (mins)")

    # time near barrier edge
    if len(session.barrier_time) > 0:
        for i, c in enumerate(tracking_data["barrier_loc"]):
            extra = 35  #
            NearBarrier = np.logical_and(
                np.logical_and(
                    tracking_data["avg_loc"][:, 0] > c[0] - extra,
                    tracking_data["avg_loc"][:, 0] < c[0] + extra,
                ),
                np.logical_and(
                    tracking_data["avg_loc"][:, 1] > c[1] - extra,
                    tracking_data["avg_loc"][:, 1] < c[1] + extra,
                ),
            )
            axs[2].plot(
                x,
                np.convolve(NearBarrier.astype(int), np.ones(session.video.fps * 60 * w), "valid")
                / (session.video.fps * 60 * w),
                color=cc(i),
            )
        axs[2].plot(
            [session.barrier_time[0], session.barrier_time[0]], [0, 1], "-k"
        )
    axs[2].set_xlabel("time (mins)")
    axs[2].legend(["left_edge", "right_edge"])
    axs[2].title.set_text("Near barrier edge")

    plt.savefig(os.path.join(save_path, "arena_occupancy_vs_time.png"))
    if settings.show_plots:
        plt.show()
    plt.close()

def hsv_hdir_colormap(angles):
    """
    Make a colormap for circular variables like hdir
    input is array of angles you need to assign a colour to"""
    phi = np.linspace(0, 2 * np.pi, len(np.arange(360)))
    rgb_cycle = np.vstack(
        (  # Three sinusoids
            0.5 * (1.0 + np.cos(phi)),  # scaled to [0,1]
            0.5 * (1.0 + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
            0.5 * (1.0 + np.cos(phi - 2 * np.pi / 3)),
        )
    ).T  # Shape = (60,3)
    bsa_rgb_cycle = np.zeros(shape=(len(angles), 3))
    for i in np.arange(360):
        bsa_rgb_cycle[angles == i + 1, :] = rgb_cycle[i, :]
    return bsa_rgb_cycle

## ------------ UNUSED FUNCTIONS
def angle_histograms(tracking_data, session, settings, save_path):
    """
    This function has been largely replaced by plot_angle_distributions
    Make histograms of head direction, head shelter angle and barrier shelter angle to ensure good sampling
    """
    figg, axs = plt.subplots(1, 3)
    figg.set_figwidth(15)

    # time in shelter (we're excluding this from our histograms)
    if "mushroom" in session.experiment:
        extra = 50  # in the mushroom session extend what the shelter is beyond the base
    else:
        extra = 0
    OutofShelterIdx = np.logical_not(
        np.logical_and(
            np.logical_and(
                tracking_data["avg_loc"][:, 0] > tracking_data["shelter_loc"][0][0] - extra,
                tracking_data["avg_loc"][:, 0] < tracking_data["shelter_loc"][1][0] + extra,
            ),
            np.logical_and(
                tracking_data["avg_loc"][:, 1] > tracking_data["shelter_loc"][0][1] - extra,
                tracking_data["avg_loc"][:, 1] < tracking_data["shelter_loc"][1][1] + extra,
            ),
        )
    )
    # head direction
    axs[0].hist(
        tracking_data["hdir"][OutofShelterIdx], np.arange(-np.pi, np.pi, np.pi / 10), density="stacked"
    )
    axs[0].set_ylabel("fraction of frames")
    axs[0].title.set_text("head dir")

    # head shelter angle
    if len(session.shelter_time) > 0:
        # only for times when there is a shelter-only
        frames_with_shelter = np.zeros_like(tracking_data["hdir_shelt"])
        if session.shelter_time[1] == -1:
            frames_with_shelter[session.shelter_time[0] * 60 * self.session.video.fps :] = 1
        else:
            frames_with_shelter[
                session.shelter_time[0] * 60
                * session.video.fps : session.shelter_time[1] * 60
                * session.video.fps
            ] = 1
        axs[1].hist(
            tracking_data["hdir_shelt"][np.logical_and(OutofShelterIdx, frames_with_shelter == 1)],
            np.arange(-np.pi, np.pi, np.pi / 10),
            density="stacked",
        )
        axs[1].title.set_text("head shelter angle")

    # head barrier angle
    if len(session.barrier_time) > 0:
        # only for times when there is a barrier
        frames_with_barrier = np.zeros_like(tracking_data["hdir_shelt"])
        if session.barrier_time[1] == -1:
            frames_with_barrier[session.barrier_time[0] * 60 * session.video.fps :] = 1
        else:
            frames_with_barrier[
                session.barrier_time[0] * 60
                * session.video.fps : session.barrier_time[1] * 60
                * session.video.fps
            ] = 1
        for c in np.arange(2):
            axs[2].hist(
                tracking_data["hdir_barrier"][np.logical_and(OutofShelterIdx, frames_with_shelter == 1), c],
                np.arange(-np.pi, np.pi, np.pi / 10),
                density="stacked",
            )
        axs[2].title.set_text("head barrier-edge angle")

    plt.savefig(os.path.join(save_path, "distribution_head_angles.png"))
    if settings.show_plots:
        plt.show()
    plt.close()