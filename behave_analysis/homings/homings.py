"""
The point of this script is to return the class Homings, the attributes of which can
be seen bellow

TODO:
- Check if escapes are being removed from homings so they don't overlap
"""

import os
from dataclasses import dataclass

import dill as pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from loguru import logger

from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration


@dataclass(frozen=True)
class Homings:
    """High level data structure for storing homings data""" ""

    onset_frames: object
    stimulus_durations: object
    fast_speed: float
    fast_angular_speed: float
    padding_duration: float
    min_change_in_dist_to_shelter: int
    max_time_within_session: float
    threat_area_height: int
    threat_area_width: int
    subgoal_locations: list


class get_Homings:
    """Extract homings from a session and then append to metadata file within session object
    
    Responsible for:
    -- Creates a Homings object
    -- Appends the Homings object to the session object
    -- Overwrites the session object"""

    def __init__(self, settings, session):
        self.settings = settings
        self.session = session

        self.extract_variables()
        self.identify_homing_runs()
        self.get_onset_and_duration()
        self.remove_inapplicable_runs()

        self.session.homing = Homings(
            self.onset_frames,
            self.stimulus_durations,
            self.settings.fast_speed,
            self.settings.fast_angular_speed,
            self.settings.padding_duration,
            self.settings.min_change_in_dist_to_shelter,
            self.settings.max_time_within_session,
            self.settings.threat_area_height,
            self.settings.threat_area_width,
            self.settings.subgoal_locations,
        )
        self.save_session()  # Add homings to session and save

    # --------MAIN FUNCS-----------------------------------------------
    def extract_variables(self):
        open_tracking_data(self)  # generates self.tracking_data
        self.get_homing_speed()
        self.get_homing_angle()
        self.get_homing_speed_angular()
        self.get_speed_along_y_axis()

    def identify_homing_runs(self):
        move_fast_to_shelter_or_edge = self.homing_speed > self.settings.fast_speed
        turn_fast_to_shelter_or_edge = self.homing_speed_angular > self.settings.fast_angular_speed
        move_at_all_to_shelter_or_edge = self.speed_along_y_axis > 0
        turn_at_all_to_shelter_or_edge = self.homing_speed_angular > 0

        go_fast_to_shelter_or_edge = move_fast_to_shelter_or_edge + turn_fast_to_shelter_or_edge
        go_at_all_to_shelter_or_edge = move_at_all_to_shelter_or_edge + turn_at_all_to_shelter_or_edge

        go_fast_to_shelter_or_edge_padded = self.boxcar_filter(
            go_fast_to_shelter_or_edge,
            self.settings.padding_duration * self.session.video.fps,
            +1,
            time="current",
        ).astype(bool)

        self.homing_runs_on = (go_fast_to_shelter_or_edge_padded * go_at_all_to_shelter_or_edge).astype(bool)

    def get_onset_and_duration(self):
        (
            self.onset_frames,
            self.stimulus_durations,
            self.offset_frames,
        ) = get_onset_and_duration(
            self.homing_runs_on,
            self.session,
            stim_type="spontaneous homings",
            min_frames_between_trials=2,
            data_type="frames",
        )

    def remove_inapplicable_runs(self):
        """
        NOTE - Does this function remove escapes?
        """
        change_in_distance_to_shelter = (
            self.distance_from_shelter[self.onset_frames] - self.distance_from_shelter[self.offset_frames]
        ) / self.distance_from_shelter[self.onset_frames]
        homing_run_durations = self.offset_frames - self.onset_frames + 1
        start_loc_x = self.tracking_data["avg_loc"][self.onset_frames, 0]
        start_loc_y = self.tracking_data["avg_loc"][self.onset_frames, 1]
        onset_time_in_session = self.onset_frames / self.session.video.fps / 60
        sufficient_move_toward_shelter = change_in_distance_to_shelter > self.settings.min_change_in_dist_to_shelter
        sufficient_run_duration = homing_run_durations > (self.settings.padding_duration * self.session.video.fps + 1)
        starts_in_threat_area = (start_loc_y < self.settings.threat_area_height) * (
            abs(start_loc_x - self.session.video.registration_size[0] / 2) < self.settings.threat_area_width / 2
        )
        starts_late_enough = onset_time_in_session < self.settings.max_time_within_session
        applicable_runs = sufficient_move_toward_shelter * sufficient_run_duration * starts_in_threat_area * starts_late_enough
        self.onset_frames = np.array([[onset_frame] for onset_frame in self.onset_frames[applicable_runs]])
        self.stimulus_durations = np.array([[stimulus_duration] for stimulus_duration in self.stimulus_durations[applicable_runs]])

    def save_session(self) -> None:
        """A fuction that saves the new session with the homings data"""
        meta_file = os.path.join(self.session.base_path, self.session.metadata_file)
        assert os.path.exists(meta_file), "Metadata file does not exist already, and it should"
        with open(meta_file, "wb") as dill_file:
            pickle.dump(self.session, dill_file)
        assert os.path.exists(meta_file), "Metadata file resaved incorrectly"
        logger.success("Session saved with homings data")

    # --------DATA PROCESSING FUNCS---------------------------------------
    def boxcar_filter(self, data, filter_length, sign, time="current"):
        if time == "past":
            filtered_data = (
                np.concatenate(
                    (
                        np.zeros(filter_length - 1),
                        np.convolve(data, np.ones(filter_length) * sign, mode="valid"),
                    )
                )
                / filter_length
            )
        if time == "future":
            filtered_data = (
                np.concatenate(
                    (
                        np.convolve(data, np.ones(filter_length) * sign, mode="valid"),
                        np.zeros(filter_length - 1),
                    )
                )
                / filter_length
            )
        if time == "current":
            filtered_data = (
                np.concatenate(
                    (
                        np.zeros(int(filter_length / 2 - 1)),
                        np.convolve(data, np.ones(filter_length) * sign, mode="valid"),
                        np.zeros(int(filter_length / 2)),
                    )
                )
                / filter_length
            )
        return filtered_data

    def get_homing_speed(self):
        reference_locations = [self.session.video.shelter_location] + self.settings.subgoal_locations
        speed_relative_to_reference_locations = np.zeros((len(self.tracking_data["avg_loc"][:, 0]) - 1, len(reference_locations)))
        for i, reference_location in enumerate(reference_locations):
            distance_from_reference_location = (
                (self.tracking_data["avg_loc"][:, 0] - reference_location[0]) ** 2
                + (self.tracking_data["avg_loc"][:, 1] - reference_location[1]) ** 2
            ) ** 0.5
            speed_pixel_per_frame = -np.diff(distance_from_reference_location)
            speed_relative_to_reference_locations[:, i] = speed_pixel_per_frame

        homing_speed_pixel_per_frame = np.max(speed_relative_to_reference_locations, axis=1)
        homing_speed_cm_per_sec = homing_speed_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_homing_speed_cm_per_sec = gaussian_filter1d(homing_speed_cm_per_sec, sigma=self.session.video.fps / 2)

        # Commenting out because looks like we removed this from the metadata file
        self.distance_from_shelter = self.tracking_data["distance rel. to shelter"]
        self.homing_speed = np.concatenate((np.zeros(1), smoothed_homing_speed_cm_per_sec))

    def get_homing_angle(self):
        reference_locations = [self.session.video.shelter_location] + self.settings.subgoal_locations
        angle_relative_to_reference_locations = np.zeros((len(self.tracking_data["avg_loc"][:, 0]), len(reference_locations)))
        for i, reference_location in enumerate(reference_locations):
            angle_relative_to_reference_locations[:, i] = np.degrees(
                np.arctan2(
                    self.tracking_data["upper_body_loc"][:, 1] - self.tracking_data["lower_body_loc"][:, 1],
                    self.tracking_data["upper_body_loc"][:, 0] - self.tracking_data["lower_body_loc"][:, 0],
                )
                - np.arctan2(
                    reference_location[1] - self.tracking_data["lower_body_loc"][:, 1],
                    reference_location[0] - self.tracking_data["lower_body_loc"][:, 0],
                )
            )
            angle_relative_to_reference_locations[:, i][angle_relative_to_reference_locations[:, i] < -180] = (
                angle_relative_to_reference_locations[:, i][angle_relative_to_reference_locations[:, i] < -180] + 360
            )
            angle_relative_to_reference_locations[:, i][angle_relative_to_reference_locations[:, i] > 180] = (
                angle_relative_to_reference_locations[:, i][angle_relative_to_reference_locations[:, i] > 180] - 360
            )

        self.shelter_angle = abs(angle_relative_to_reference_locations[:, 0])
        self.homing_angle = np.min(abs(angle_relative_to_reference_locations), axis=1)

    def get_homing_speed_angular(self):
        angular_speed_deg_per_frame = -np.diff(self.homing_angle)
        angular_speed_deg_per_sec = angular_speed_deg_per_frame * self.session.video.fps
        smoothed_angular_speed_deg_per_sec = gaussian_filter1d(angular_speed_deg_per_sec, sigma=self.session.video.fps / 10)
        self.homing_speed_angular = np.concatenate((np.zeros(1), smoothed_angular_speed_deg_per_sec))

    def get_speed_along_y_axis(self):
        speed_y_pixel_per_frame = np.diff(self.tracking_data["avg_loc"][:, 1], axis=0)
        speed_y_cm_per_sec = speed_y_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_speed_y_cm_per_sec = gaussian_filter1d(speed_y_cm_per_sec, sigma=self.session.video.fps / 10)
        self.speed_along_y_axis = np.concatenate((np.zeros(1), smoothed_speed_y_cm_per_sec))
