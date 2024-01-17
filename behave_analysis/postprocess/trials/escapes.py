"""
Extract information about escapes in a session
"""

import os
from dataclasses import dataclass

import dill as pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from behave_analysis.visualize.visualize_utils import open_tracking_data
from loguru import logger
from behave_analysis.analyze.behaviour.spatial_efficiency import plot_escape_trajectories, plot_optimal_trajectories, identify_condition_escape


@dataclass(frozen=True)
class Escapes:
    """Data object for storing escape information"""

    stim_onset_frames: list  # when was the stim presented
    stimulus_durations: list
    escape_onset_frames: list  # when did the actual escape start
    head_orientation: list
    spatial_efficiency: list
    escape_condition: list  # the condition the escape happened in e.g. 'shelter_only', 'barrier_pre_flip'
    trajectory_length: list
    optimal_trajectory_length: list
    spatial_efficiency: list
    # these are homings attributes: do we want them for escape also?
    # fast_speed: float
    # fast_angular_speed: float
    # padding_duration: float
    # min_change_in_dist_to_shelter: int
    # max_time_within_session: float


class get_Escapes:
    """Extract information about escapes from a session and sves it.
    This will be called from postprocess.

    Responsible for:
    -- Creates an Escape object"""

    def __init__(self, settings, session, tracking_data, video_df):
        onset_frames = session.__dict__[settings.stim_type].onset_frames
        stimulus_durations = session.__dict__[settings.stim_type].stimulus_duration

        # find escape onset
        esc_onset = onset_frames

        # spatial efficiency
        condition, trajectory_length, optimal_trajectory_length, spatial_efficiency = get_spatial_efficiency(
            settings, session, tracking_data, video_df
        )

        self.session.homing = Escapes(
            stim_onset_frames=onset_frames,
            stimulus_durations=stimulus_durations,
            escape_onset_frames=esc_onset,
            escape_condition=condition,
            trajectory_length=trajectory_length,
            optimal_trajectory_length = optimal_trajectory_length,
            spatial_efficiency=spatial_efficiency,
            # subgoal_locations: list
            # head_orientation: list
            # spatial_efficiency: list
        )
        self.save_session()  # Add homings to session and save


def get_spatial_efficiency(onset_frames, stimulus_durations, session, tracking_data, video_df):
    condition = np.empty(len(onset_frames))
    trajectory_length = np.empty(len(onset_frames))
    optimal_trajectory_length = np.empty(len(onset_frames))
    spatial_efficiency_value = np.empty(len(onset_frames))
    for trial_num, (on_fr, st) in enumerate(zip(onset_frames, stimulus_durations)):
        condition = identify_condition_escape(video_df.filter(video_df["frames"] == onset_frames), session)
        trajectory_length[trial_num] = plot_escape_trajectories(on_fr[0], stimulus_durations[0] * session.video.fps, tracking_data)
        optimal_trajectory_length[trial_num] = plot_optimal_trajectories(on_fr[0], tracking_data, condition)
        spatial_efficiency_value[trial_num] = optimal_trajectory_length[trial_num] / trajectory_length[trial_num]
