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
from behave_analysis.utils.identify_condition import identify_condition_of_trial
from behave_analysis.visualize.visualize_utils import open_tracking_data

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
        

    def initialize_dict(self, onset_frames, stimulus_durations):
        return {
            "stim_onset_frames": list(onset_frames), # when was the stim presented
            "stimulus_durations": list(stimulus_durations),
            "onset_frames": list(onset_frames), # when did the actual escape start
            "offset_frames": [a + b for a, b in zip(onset_frames, stimulus_durations)],
            "escape_latency_sec": np.zeros_like(onset_frames).astype(float), # how many seconds after stim onset did the mouse escape
            "freeze_bool": np.zeros_like(onset_frames), # did the mouse freeze?
            "start_locs": np.zeros((len(onset_frames), 2)), # x,y pixel locations of the start of each homing run
            "end_locs": np.zeros((len(onset_frames), 2)), # x,y pixel locations of the end of each homing run
            "avg_speed": np.zeros_like(onset_frames, dtype=float), # Average speed in cm/s across homing
            "head_orientation_dic": {}, # In the first 15cm of the homing run, avg angle to reference locations
            "hdir_at_start": np.zeros_like(onset_frames, dtype=float),
            "condition": [], # the condition the escape happened in e.g. 'shelter_only', 'barrier_pre_flip'
            "trajectory_length": [], # how long the path of each escape was
            "spatial_efficiency": [],
        }

    def get_escape(self):
        onset_frames = self.session.__dict__[self.settings.escape_stim_type].onset_frames
        stimulus_durations = self.session.__dict__[self.settings.escape_stim_type].stimulus_durations
        
        if len(onset_frames) > 0:
            if isinstance(onset_frames[0], np.ndarray):
                onset_frames = [on[0] for on in onset_frames]
                stimulus_durations = [st[0] for st in stimulus_durations]

        self.results = self.initialize_dict(onset_frames, stimulus_durations)

        if len(onset_frames) == 0:
            logger.warning("No escapes in this session!")
            self.save_session()  # save escapes to pickle
            return

        head_orientation_dic = {}
        for key in self.homings["head_orientation_dic"].keys():
            if key not in head_orientation_dic:
                head_orientation_dic[key] = []

        # find escape onset
        for c_fr, on_fr in enumerate(onset_frames):
            # this is always the head ori when the stim was first played
            self.results["hdir_at_start"][c_fr] = self.tracking_data["hdir"][on_fr]
            self.homings["escapes"] = np.zeros_like(self.homings["onset_frames"], dtype=bool)  # init a new column in the homings object to mark which homings are escapes

            if len(self.homings["onset_frames"]) > 0:
                (
                    self.results["offset_frames"][c_fr],
                    self.results["start_locs"][c_fr, :],
                    self.results["end_locs"][c_fr, :],
                    self.results["avg_speed"][c_fr],
                    head_orientation_dic,
                    self.results["onset_frames"][c_fr],
                    self.results["escape_latency_sec"][c_fr],
                    self.homings,
                ) = check_if_in_homing_obj(self.homings, on_fr, self.settings, self.session, head_orientation_dic)

            # if no homing after escape, did the mouse freeze?
            if self.results["onset_frames"][c_fr] == 0:  # that means that it wasn't a homing
                # where was mousie when stim turned on?
                self.results["start_locs"][c_fr, :], self.results["end_locs"][c_fr, :] = get_start_and_end_locs(
                    tracking=self.tracking_data,
                    onset_frames=[on_fr],
                    offset_frames=[on_fr],
                )

                # did he run?
                (
                    self.results["onset_frames"][c_fr],
                    self.results["offset_frames"][c_fr],
                    ht,
                    self.results["avg_speed"][c_fr],
                ) = escape_or_freeze(
                    tracking_data=self.tracking_data,
                    on_fr=on_fr,
                    session=self.session,
                    settings=self.settings,
                    fps=self.session.video.fps,
                    angles=head_orientation_dic.keys(),
                )
                for key in self.homings["head_orientation_dic"].keys():
                    head_orientation_dic[key].append(ht[key])
                if np.isnan(self.results["onset_frames"][c_fr]):
                    self.results["escape_latency_sec"][c_fr] = np.nan
                    self.results["freeze_bool"][c_fr] = 1
                else:
                    self.results["escape_latency_sec"][c_fr] = (self.results["onset_frames"][c_fr] - on_fr) / self.session.video.fps

            self.results["condition"].append(identify_condition_of_trial(self.video_df.filter(self.video_df["frames"] == int(on_fr)), self.session))

        self.results["spatial_efficiency"], self.results["trajectory_length"] = spatial_efficiency(
            onset_frames,
            stimulus_durations,
            self.session,
            self.settings,
            self.results["condition"],
            self.tracking_data,
            trial_type="Escapes",
            plotting=False,
        )

        self.results["head_orientation_dic"] = head_orientation_dic

        self.save_session()  # save escapes to pickle

        return self.results, self.homings

    def save_session(self) -> None:
        """Save ecape object as a pickle file within the session folder"""
        folder = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "escapes"))
        file_name = os.path.join(folder, "escapes.npy")
        np.save(file_name, self.results, allow_pickle=True)
        logger.success("Escape dict saved")


def check_if_in_homing_obj(homings, on_fr, settings, session, head_theta):
    h_idx = np.argmin(np.abs(homings["onset_frames"] - on_fr))
    h_nearest_to_stim = homings["onset_frames"][h_idx]
    # find if there is a homing right after the stim
    # DEF: it must start after the stim and within 5 seconds of stim
    if np.logical_or(
        np.logical_and(
            (h_nearest_to_stim - on_fr) > 0,
            (h_nearest_to_stim - on_fr) <= (settings.escape_response_thresh * session.video.fps),
        ),
        np.logical_and((h_nearest_to_stim - on_fr) < 0, homings["offset_frames"][h_idx] > on_fr),
    ):

        esc_offset = homings["offset_frames"][h_idx]
        start_locs = homings["start_locs"][h_idx]
        end_locs = homings["end_locs"][h_idx]
        avg_speed = homings["avg_speed"][h_idx]
        for key in homings["head_orientation_dic"].keys():
            head_theta[key].append(homings["head_orientation_dic"][key][h_idx])

        if np.logical_and(
            (h_nearest_to_stim - on_fr) > 0,
            (h_nearest_to_stim - on_fr) <= (settings.escape_response_thresh * session.video.fps),
        ):
            esc_onset = h_nearest_to_stim
            esc_latency = float((h_nearest_to_stim - on_fr) / session.video.fps)  # in seconds

        # find if there is a homing started right before the stim
        # DEF: the homing must start before and finish after the stim (no time constraint)
        elif np.logical_and((h_nearest_to_stim - on_fr) < 0, homings["offset_frames"][h_idx] > on_fr):
            esc_onset = on_fr
            esc_latency = 0  # (on_fr) / session.video.fps  # in seconds

        # flag this homing as an escape!
        homings["escapes"][h_idx] = True

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
    fast_mousie = np.hstack((np.zeros(fps), mousie_speed > settings.escape_speed_threshold))
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
                settings.homings_distance_threshold,
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
