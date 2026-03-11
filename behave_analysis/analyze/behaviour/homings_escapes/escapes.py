"""
Extract information about escapes in a session
"""

import os
from dataclasses import dataclass

import dill as pickle
import numpy as np
import polars as pl
from loguru import logger
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.behaviour.homings_escapes.homings import (
    get_avg_homing_angle_for_first15cm_of_run,
    get_start_and_end_locs,
    get_avg_speed,
)
from behave_analysis.analyze.behaviour.utils import identify_condition_of_trial
from behave_analysis.visualize.visualize_utils import open_tracking_data


@dataclass(frozen=True)
class Escapes:
    """Data object for storing escape information"""

    stim_onset_frames: list  # when was the stim presented
    stimulus_durations: list
    onset_frames: list  # when did the actual escape start
    offset_frames: list
    escape_latency_sec: list  # how many seconds after stim onset did the mouse escape
    freeze_bool: list  # did the mouse freeze?
    start_locs: np.array  # x,y pixel locations of the start of each homing run
    end_locs: np.array  # x,y pixel locations of the end of each homing run
    avg_speed: np.array  # Average speed in cm/s across homing
    head_orientation_dic: dict  # In the first 15cm of the homing run, avg angle to reference locations
    hdir_at_start: np.array
    condition: list  # the condition the escape happened in e.g. 'shelter_only', 'barrier_pre_flip'
    trajectory_length: list  # how long the path of each escape was
    # optimal_trajectory_length: list
    spatial_efficiency: list


class get_Escapes:
    """Extract information about escapes from a session and sves it.
    This will be called from postprocess.

    Responsible for:
    -- Creates an Escape object

    Escapes are defined by what happens after the stimulus.
    If an 'escape' is found in the homings object, it will be removed from the homings object
    """

    def __init__(self, settings, session, tracking_data=[], video_df=[], homings=None):
        self.settings = settings
        self.session = session
        self.homings = homings
        self.tracking_data = tracking_data
        self.video_df = video_df
        if len(tracking_data) == 0:
            self.tracking_data = open_tracking_data(session)
        if len(video_df) == 0:
            self.video_df = pl.read_csv(os.path.join(session.base_path, session.processed_path) + "\\" "full_video_dataframe.csv")

    def get_escape(self):
        onset_frames = self.session.__dict__[self.settings.stim_type].onset_frames
        stimulus_durations = self.session.__dict__[self.settings.stim_type].stimulus_durations
        if len(onset_frames) == 0:
            logger.warning("No escapes in this session!")
            self.escapes = Escapes(
                stim_onset_frames=[],
                stimulus_durations=[],
                onset_frames=[],
                offset_frames=[],
                escape_latency_sec=[],
                freeze_bool=[],
                condition=[],  # what condition did the escape happen in e.g. 'shelter_only'
                trajectory_length=[],
                spatial_efficiency=[],
                head_orientation_dic=[],
                start_locs=[],
                end_locs=[],
                avg_speed=[],
                hdir_at_start=[],
            )
            self.save_session(self.session)  # save escapes to pickle
            return

        if isinstance(onset_frames[0], np.ndarray):
            onset_frames = [on[0] for on in onset_frames]
            stimulus_durations = [st[0] for st in stimulus_durations]

        # init varsq
        esc_onset = list(onset_frames)  # when did the actual escape start
        esc_offset = [a + b for a, b in zip(onset_frames, stimulus_durations)]  # when did the actual escape start
        esc_latency = np.zeros_like(onset_frames).astype(float)  # how many seconds after stim onset did the mouse escape
        freeze = np.zeros_like(onset_frames)  # did the mouse freeze?
        start_locs = np.zeros((len(onset_frames), 2))
        end_locs = np.zeros((len(onset_frames), 2))
        avg_speed = np.zeros_like(onset_frames, dtype=float)
        head_ori_at_start = np.zeros_like(onset_frames, dtype=float)
        head_theta = {}
        condition = []
        for key in self.homings.head_orientation_dic.keys():
            if key not in head_theta:
                head_theta[key] = []

        # find escape onset
        for c_fr, on_fr in enumerate(onset_frames):
            # this is always the head ori when the stim was first played
            head_ori_at_start[c_fr] = self.tracking_data["hdir"][on_fr]

            (
                esc_offset[c_fr],
                start_locs[c_fr, :],
                end_locs[c_fr, :],
                avg_speed[c_fr],
                head_theta,
                esc_onset[c_fr],
                esc_latency[c_fr],
                self.homings,
            ) = check_if_in_homing_obj(self.homings, on_fr, self.settings, self.session, head_theta)

            # if no homing after escape, did the mouse freeze?
            if esc_onset[c_fr] == 0:  # that means that it wasn't a homing
                # where was mousie when stim turned on?
                start_locs[c_fr, :], end_locs[c_fr, :] = get_start_and_end_locs(
                    tracking=self.tracking_data,
                    onset_frames=[on_fr],
                    offset_frames=[on_fr],
                )

                # did he run?
                (
                    esc_onset[c_fr],
                    esc_offset[c_fr],
                    ht,
                    avg_speed[c_fr],
                ) = escape_or_freeze(
                    tracking_data=self.tracking_data,
                    on_fr=on_fr,
                    session=self.session,
                    settings=self.settings,
                    fps=self.session.video.fps,
                    angles=head_theta.keys(),
                )
                for key in self.homings.head_orientation_dic.keys():
                    head_theta[key].append(ht[key])
                if np.isnan(esc_onset[c_fr]):
                    esc_latency[c_fr] = np.nan
                    freeze[c_fr] = 1
                else:
                    esc_latency[c_fr] = (esc_onset[c_fr] - on_fr) / self.session.video.fps

            condition.append(identify_condition_of_trial(self.video_df.filter(self.video_df["frames"] == int(on_fr)), self.session))

        spatial_efficiency_values, trajectory_length = spatial_efficiency(
            onset_frames,
            stimulus_durations,
            self.session,
            self.settings,
            condition,
            self.tracking_data,
            trial_type="Escapes",
            plotting=False,
        )

        self.escapes = Escapes(
            stim_onset_frames=onset_frames,
            stimulus_durations=stimulus_durations,
            onset_frames=esc_onset,
            offset_frames=esc_offset,
            escape_latency_sec=esc_latency,
            freeze_bool=freeze,
            condition=condition,  # what condition did the escape happen in e.g. 'shelter_only'
            trajectory_length=trajectory_length,
            spatial_efficiency=spatial_efficiency_values,
            head_orientation_dic=head_theta,
            start_locs=start_locs,
            end_locs=end_locs,
            avg_speed=avg_speed,
            hdir_at_start=head_ori_at_start,
        )

        self.save_session(self.session)  # save escapes to pickle

        return self.escapes

    def save_session(self, session) -> None:
        """Save ecape object as a pickle file within the session folder"""
        folder = make_directory(os.path.join(session.base_path, session.processed_path, "escapes"))
        file_name = os.path.join(folder, "escapes_obj.pkl")
        with open(file_name, "wb") as dill_file:
            pickle.dump(self.escapes, dill_file)
        logger.success("Escape pickle object saved")


def check_if_in_homing_obj(homings, on_fr, settings, session, head_theta):
    h_nearest_to_stim = homings.onset_frames[np.argmin(np.abs(homings.onset_frames - on_fr))]
    h_idx = int(np.where(homings.onset_frames == h_nearest_to_stim)[0])
    # find if there is a homing right after the stim
    # DEF: it must start after the stim and within 5 seconds of stim
    if np.logical_or(
        np.logical_and(
            (h_nearest_to_stim - on_fr) > 0,
            (h_nearest_to_stim - on_fr) <= (settings.response_thresh * session.video.fps),
        ),
        np.logical_and((h_nearest_to_stim - on_fr) < 0, homings.offset_frames[h_idx] > on_fr),
    ):

        esc_offset = homings.offset_frames[h_idx]
        start_locs = homings.start_locs[h_idx]
        end_locs = homings.end_locs[h_idx]
        avg_speed = homings.avg_speed[h_idx]
        for key in homings.head_orientation_dic.keys():
            head_theta[key].append(homings.head_orientation_dic[key][h_idx])

        if np.logical_and(
            (h_nearest_to_stim - on_fr) > 0,
            (h_nearest_to_stim - on_fr) <= (settings.response_thresh * session.video.fps),
        ):
            esc_onset = h_nearest_to_stim
            esc_latency = float((h_nearest_to_stim - on_fr) / session.video.fps)  # in seconds

        # find if there is a homing started right before the stim
        # DEF: the homing must start before and finish after the stim (no time constraint)
        elif np.logical_and((h_nearest_to_stim - on_fr) < 0, homings.offset_frames[h_idx] > on_fr):
            esc_onset = on_fr
            esc_latency = 0  # (on_fr) / session.video.fps  # in seconds

        # remove this homing from the homing object because it's an escape!
        for key in homings.__dict__:
            if key == "head_orientation_dic":
                for key in homings.head_orientation_dic.keys():
                    homings.head_orientation_dic[key] = np.delete(homings.head_orientation_dic[key], h_idx)
            else:
                if isinstance(homings.__dict__[key], list):
                    del homings.__dict__[key][h_idx]
                elif isinstance(homings.__dict__[key], np.ndarray):
                    homings.__dict__[key] = np.delete(homings.__dict__[key], h_idx, 0)

        return (
            esc_offset,
            start_locs,
            end_locs,
            avg_speed,
            head_theta,
            esc_onset,
            esc_latency,
            homings,
        )
    else:  # no homing after stim
        return (0, [0,0], [0,0], 0, head_theta, 0, 0, homings)  # no escape found


def escape_or_freeze(tracking_data, on_fr, session, settings, fps, angles):
    """A function that looks at the behaviour of mousie after stim"""
    # init vars
    head_theta = {}
    for a in angles:
        head_theta[a] = np.nan

    # what is mouse speed in 20s following stim
    mousie_speed = tracking_data["avg_Velocity"][int(on_fr) : int(on_fr) + (20 * session.video.fps)]

    # find escape
    fast_mousie = np.hstack((np.zeros(fps), mousie_speed > settings.fast_speed))
    run_onset = np.diff(
        np.convolve(
            fast_mousie,
            np.hstack((np.zeros(int(fps / 4)), np.ones(int(fps / 4)))),
            mode="same",
        )
    )
    if len(np.where(run_onset)[0]) > 0:  # mousie needs to run at 15cm/s for a few consecutive frames
        esc_onset = np.where(run_onset)[0][0] - fps  # which frame does escape start on? the stim is frame 0
        esc_offset = np.where(run_onset == -1)[0][0] - fps

        if (esc_offset - esc_onset) > fps / 2:  # escape needs to last at least .5 sec?
            # get escape properties - these functions want lists!!!
            avg_speed = get_avg_speed([esc_onset + on_fr], [esc_offset + on_fr], tracking_data, session)
            head_theta, _ = get_avg_homing_angle_for_first15cm_of_run(
                session,
                [on_fr],
                [esc_offset + on_fr],
                tracking_data,
                settings.cum_threshold,
            )
            esc_onset = esc_onset + on_fr
            esc_offset = esc_offset + on_fr
        else:  # ok this is a freeze/no escpae situation
            esc_onset = np.nan
            esc_offset = np.nan
            avg_speed = 0

    return (
        esc_onset,
        esc_offset,
        head_theta,
        avg_speed,
    )


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
