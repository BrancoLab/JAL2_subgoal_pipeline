"""
Extract information about escapes in a session
"""

import os
from dataclasses import dataclass

import dill as pickle
import numpy as np
from loguru import logger
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.homings.homings import get_avg_homing_angle_for_start_of_run
from settings.settings_analyze import settings_analyze as settings_a
from settings.settings_homings import settings_homings as settings_h


@dataclass(frozen=True)
class Escapes:
    """Data object for storing escape information"""

    stim_onset_frames: list  # when was the stim presented
    stimulus_durations: list
    escape_onset_frames: list  # when did the actual escape start
    escape_latency: list  # how many seconds after stim onset did the mouse escape
    freeze_bool: list  # did the mouse freeze?
    head_orientation: list
    escape_condition: list  # the condition the escape happened in e.g. 'shelter_only', 'barrier_pre_flip'
    trajectory_length: list
    optimal_trajectory_length: list
    spatial_efficiency: list


class get_Escapes:
    """Extract information about escapes from a session and sves it.
    This will be called from postprocess.

    Responsible for:
    -- Creates an Escape object"""

    def __init__(self, settings, session, tracking_data, video_df, homings):
        onset_frames = session.__dict__[settings_a.stim_type].onset_frames
        stimulus_durations = session.__dict__[settings_a.stim_type].stimulus_durations

        # init varsq
        esc_onset = np.zeros_like(onset_frames)  # when did the actual escape start
        esc_latency = np.zeros_like(onset_frames)  # how many seconds after stim onset did the mouse escape
        freeze = np.zeros_like(onset_frames)  # did the mouse freeze?
        head_theta = {}
        for key in homings.homing_angles_dic.keys():
            if key not in head_theta:
                head_theta[key] = []

        # find escape onset
        for c_fr, on_fr in enumerate(onset_frames):
            h_nearest_to_stim = homings.onset_frames[np.argmin(np.abs(homings.onset_frames - on_fr))]
            # find if there is a homing right after the stim
            # DEF: it must start after the stim and within 5 seconds of stim
            if np.logical_and((h_nearest_to_stim - on_fr) > 0, (h_nearest_to_stim - on_fr) <= (settings.response_thresh * session.video.fps)):
                esc_onset[c_fr] = h_nearest_to_stim
                esc_latency[c_fr] = (h_nearest_to_stim - on_fr) / session.video.fps  # in seconds
                for key in homings.homing_angles_dic.keys():
                    head_theta[key].append(homings.homing_angles_dic[key][int(np.where(homings.onset_frames == h_nearest_to_stim)[0])])

            # find if there is a homing started right before the stim
            # DEF: the homing must start before and finish after the stim (no time constraint)
            elif np.logical_and(
                (h_nearest_to_stim - on_fr) < 0, homings.offset_frames[np.where(homings.onset_frames == h_nearest_to_stim)[0]] > on_fr
            ):
                esc_onset[c_fr] = on_fr
                esc_latency[c_fr] = (on_fr) / session.video.fps  # in seconds
                for key in homings.homing_angles_dic.keys():
                    head_theta[key].append(homings.homing_angles_dic[key][int(np.where(homings.onset_frames == h_nearest_to_stim)[0])])
            # if no homing after escape, did the mouse freeze?
            else:
                esc_onset[c_fr], ht = escape_or_freeze(tracking_data, on_fr, session, settings_h, session.video.fps, angles=head_theta.keys())
                for key in homings.homing_angles_dic.keys():
                    head_theta[key].append(ht[key])
                if np.isnan(esc_onset[c_fr]):
                    esc_latency[c_fr] = np.nan
                    freeze[c_fr] = 1
                else:
                    esc_latency[c_fr] = (esc_onset[c_fr] - on_fr) / session.video.fps

        # spatial efficiency
        condition, trajectory_length, optimal_trajectory_length, spatial_efficiency_values = spatial_efficiency(
            onset_frames, stimulus_durations, session, settings, tracking_data, video_df, plotting = False
        )

        self.escapes = Escapes(
            stim_onset_frames=onset_frames,
            stimulus_durations=stimulus_durations,
            escape_onset_frames=esc_onset,
            escape_latency=esc_latency,
            freeze_bool=freeze,
            escape_condition=condition,  # what condition did the escape happen in e.g. 'shelter_only'
            trajectory_length=trajectory_length,
            optimal_trajectory_length=optimal_trajectory_length,
            spatial_efficiency=spatial_efficiency_values,
            head_orientation=head_theta,
        )

        self.save_session()  # save escapes to pickle

    def save_session(self) -> None:
        """Save ecape object as a pickle file within the session folder"""
        folder = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "escapes"))
        file_name = os.path.join(folder, "escapes_obj.pkl")
        with open(file_name, "wb") as dill_file:
            pickle.dump(self.escapes, dill_file)
        logger.success("Escape pickle object saved")


def escape_or_freeze(tracking_data, on_fr, session, settings_h, fps, angles):
    """A function that looks at the behaviour of mousie after stim"""
    esc_onset = np.nan
    head_theta = {}
    for a in angles:
        head_theta[a] = np.nan
    # what is mouse speed in 20s following stim
    mousie_speed = tracking_data["avg_Velocity"][int(on_fr) : int(on_fr) + (20 * session.video.fps)]

    # find escape
    fast_mousie = np.hstack((np.zeros(fps), mousie_speed > settings_h.fast_speed))
    run_onset = np.diff(np.convolve(fast_mousie, np.hstack((np.zeros(int(fps / 4)), np.ones(int(fps / 4)))), mode="same"))
    if len(np.where(run_onset)[0]) > 0:  # mousie needs to run at 15cm/s for a few consecutive frames
        esc_onset = np.where(run_onset)[0][0] - fps
        esc_offset = np.where(run_onset == -1)[0][0] - fps
        if (esc_offset - esc_onset) > fps / 2:  # escape needs to last at least .5 sec?
            # get head angles
            head_theta = get_avg_homing_angle_for_start_of_run(
                session, esc_onset + on_fr, esc_offset + on_fr, tracking_data, settings_h.cum_threshold
            )

    return esc_onset, head_theta


# def get_spatial_efficiency(onset_frames, stimulus_durations, session, tracking_data, video_df):
#     condition = []
#     trajectory_length = np.empty(len(onset_frames))
#     optimal_trajectory_length = np.empty(len(onset_frames))
#     spatial_efficiency_value = np.empty(len(onset_frames))
#     for trial_num, (on_fr, st) in enumerate(zip(onset_frames, stimulus_durations)):
#         condition.append([identify_condition_escape(video_df.filter(video_df["frames"] == on_fr[0]), session)])
#         trajectory_length[trial_num] = plot_escape_trajectories(on_fr[0], st[0] * session.video.fps, tracking_data)
#         optimal_trajectory_length[trial_num] = plot_optimal_trajectories(on_fr[0], tracking_data, condition[trial_num][0])
#         spatial_efficiency_value[trial_num] = optimal_trajectory_length[trial_num] / trajectory_length[trial_num]
#     return condition, trajectory_length, optimal_trajectory_length, spatial_efficiency_value
