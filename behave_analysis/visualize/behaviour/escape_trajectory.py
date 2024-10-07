'''A set of functions for visualizing the escape trajectories of a mouse in a given session'''

# set up
import os
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import
# from behave_analysis.analyze.behaviour.spatial_efficiency import identify_condition_escape, base_plotting
from behave_analysis.analyze.behaviour.utils import base_plotting, identify_condition_of_trial
from behave_analysis.utils.arena_plotting import Arena

def escape_trajectory_and_shelter_exits(tracking_data, video_df, stim_type, session, settings, save_path):
    """
    Plot escape trajectories as well as the path by which the mouse last left the shelter
    """

    # set up figure and number of rows and calculate number of columns
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3)
    ntrial = len(session.__dict__[stim_type].onset_frames)
    nrows = 3
    ncols = ntrial // nrows + (ntrial % nrows > 0)

    for trial_num, (onset_frames, stimulus_durations) in enumerate(
        zip(
            session.__dict__[stim_type].onset_frames,
            session.__dict__[stim_type].stimulus_durations,
        )
    ):
        ax = plt.subplot(nrows, ncols, trial_num + 1)
        # set up axes with shelt and barrier locations
        condition = identify_condition_of_trial(video_df.filter(video_df["frames"] == onset_frames), session)
        Arena(ax=ax, shelter_coordinates=tracking_data["shelter_loc"], condition=condition, barrier_coordinates=session.barrier_location)
        # base_plotting(ax, tracking_data, condition, session = session)
        # plot escape trajectory
        plot_trajectories(
            tracking_data["head_loc"],
            tracking_data["avg_Velocity"],
            onset_frames[0],
            stimulus_durations[0],
            ax,
        )
        # plot shelter exit trajectory
        exits = np.where(np.diff(video_df["OutofshelterIdx"].to_numpy().astype(int)) == 1)[0]
        exits = exits[exits < onset_frames[0]]
        plot_trajectories(
            tracking_data["head_loc"],
            tracking_data["avg_Velocity"],
            exits[-1] - session.video.fps,
            3,
            ax,
            colors="Blues",
        )

    filename = str(save_path) + "/" + "Escape_and_Exit" + ".png"
    plt.savefig(filename)
    if settings.show_plots:
        plt.show()
    plt.close()

def plot_trajectories(head_loc, velocity, onset_frames, stimulus_durations, ax, colors="Reds"):
    """
    Plot escape trajectories
    """
    # compute and plot each trajectory
    x_loc = head_loc[onset_frames : onset_frames + int(stimulus_durations * 40), 0]
    y_loc = head_loc[onset_frames : onset_frames + int(stimulus_durations * 40), 1]
    speed = velocity[onset_frames : onset_frames + int(stimulus_durations * 40)]
    ax.scatter(x_loc, y_loc, s=5, c=speed, cmap=colors)