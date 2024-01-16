"""
The point of this script is to return the class Homings, the attributes of which can
be seen below. Escapes are not removed from homings here currently. And need to be removed elsewhere
potentially post processing?

BUG:
-- Sometimes the meta file is not saved correctly and process has to be rerun
"""
from statistics import mean
import os
from dataclasses import dataclass

from loguru import logger
import dill as pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d

from behave_analysis.visualize.visualize_utils import open_tracking_data
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
    -- Overwrites the session object

    Misc:
    -- Reference locations = [shelter, subgoal1, subgoal2] ignoring central barrier location
    """

    def __init__(self, settings, session):
        self.settings = settings
        self.session = session
        self.reference_locations = [self.session.video.shelter_location] + self.session.barrier_location[:-1]

        self.extract_variables()
        self.homing_runs_on = self.identify_homing_runs()
        self.onset_frames, self.stimulus_durations, self.offset_frames = self.get_onset_and_duration()
        self.onset_frames, self.stimulus_durations = self.remove_inapplicable_runs()

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
            self.session.barrier_location[:-1],
        )
        self.save_session()  # Add homings to session and save

    # --------MAIN FUNCS-----------------------------------------------
    def extract_variables(self):
        """Extract the variables needed to identify homings"""
        self.tracking_data = open_tracking_data(self.session)
        self.homing_speed = self.get_homing_speed()
        self.homing_angle = self.get_homing_angle()
        self.homing_speed_angular = self.get_homing_speed_angular()
        self.speed_along_y_axis = self.get_speed_along_y_axis()

    def identify_homing_runs(self):
        """Using threshold criteria, identify homing runs

        Criteria:
        -- Speed must be above a threshold
        -- Head must turn towards shelter or edge at a threshold
        -- Speed along y axis must be positive
        -- Angular speed must be positive"""

        # booleans for whether each frame meets threshold criteria
        move_fast_to_shelter_or_edge = self.homing_speed > self.settings.fast_speed
        turn_fast_to_shelter_or_edge = self.homing_speed_angular > self.settings.fast_angular_speed
        move_at_all_to_shelter_or_edge = self.speed_along_y_axis > 0
        turn_at_all_to_shelter_or_edge = self.homing_speed_angular > 0

        # add boolen arrya toi perform an OR operation
        go_fast_to_shelter_or_edge = move_fast_to_shelter_or_edge + turn_fast_to_shelter_or_edge
        go_at_all_to_shelter_or_edge = move_at_all_to_shelter_or_edge + turn_at_all_to_shelter_or_edge

        # Return a boolean array of shape (filter_length, )
        go_fast_to_shelter_or_edge_padded = self.boxcar_filter(
            data=go_fast_to_shelter_or_edge,
            filter_length=self.settings.padding_duration * self.session.video.fps,
            sign=+1,
            time="current",
        ).astype(bool)

        # multiply the two bool arrays to perform an AND operation
        homing_runs_on = (go_fast_to_shelter_or_edge_padded * go_at_all_to_shelter_or_edge).astype(bool)

        return homing_runs_on

    def get_onset_and_duration(self):
        """Code to get the onset and duration of homing runs

        Returns:
        -- onset_frames (np.array): Each homing onset in frames
        -- stimulus_durations (np.array): Each homing duration in seconds
        -- offset_frames (np.array): Each homing offset in frames
        """
        (
            onset_frames,
            stimulus_durations,
            offset_frames,
        ) = get_onset_and_duration(
            self.homing_runs_on,
            self.session,
            stim_type="spontaneous homings",
            min_frames_between_trials=2,
            data_type="frames",
        )

        return onset_frames, stimulus_durations, offset_frames

    def remove_inapplicable_runs(self) -> tuple[np.array, np.array]:
        """
        Remove homings that don't meet the criteria set in settings_homings.py

        Homings must:
        -- Move towards the shelter or subgoal
        -- Be long enough to be considered a homing run
        -- Start in the threat area or near a subgoal
        """
        self.distance_from_shelter = self.tracking_data["distance rel. to shelter"]

        # Where does the homing run start?
        start_loc_x = self.tracking_data["avg_loc"][self.onset_frames, 0]
        start_loc_y = self.tracking_data["avg_loc"][self.onset_frames, 1]

        # Does the mouse move enough towards the shelter to be considered a homing run?
        change_in_distance_to_shelter = (
            self.distance_from_shelter[self.onset_frames] - self.distance_from_shelter[self.offset_frames]
        ) / self.distance_from_shelter[self.onset_frames]
        sufficient_move_toward_shelter = change_in_distance_to_shelter > self.settings.min_change_in_dist_to_shelter

        # Is the homing run long enough to be considered a homing run?
        homing_run_durations = self.offset_frames - self.onset_frames + 1
        sufficient_run_duration = homing_run_durations > (self.settings.padding_duration * self.session.video.fps + 1)

        # Is the homing run in the threat area?
        starts_in_threat_area = (start_loc_y < self.settings.threat_area_height) * (
            abs(start_loc_x - self.session.video.registration_size[0] / 2) < self.settings.threat_area_width / 2
        )

        # Does the homing run start near one of the subgoals?
        subgoal_locations = self.reference_locations[1:]
        subgoal_locations_x = [x[0] for x in subgoal_locations]
        subgoal_locations_y = mean([x[1] for x in subgoal_locations])  # Can take average of Y as should be the same for both subgoals

        # Does the mouse x and y poistion start within 10% around the subgoal?
        start_loc_x_within_subgoal = (start_loc_x > subgoal_locations_x[0] * 0.9) * (start_loc_x < subgoal_locations_x[0] * 1.1)
        start_loc_y_within_subgoal = (start_loc_y > subgoal_locations_y * 0.9) * (start_loc_y < subgoal_locations_y * 1.1)
        starts_in_subgoal = start_loc_x_within_subgoal * start_loc_y_within_subgoal

        # Some philip logic we can ignore?
        onset_time_in_session = self.onset_frames / self.session.video.fps / 60
        starts_late_enough = onset_time_in_session < self.settings.max_time_within_session

        # Apply all the criteria using an AND operation, make in threat area or starts in subgoal
        applicable_runs = sufficient_move_toward_shelter * sufficient_run_duration * (starts_in_threat_area + starts_in_subgoal) * starts_late_enough

        # Extract the onset frames and durations of the applicable runs
        onset_frames = np.array([[onset_frame] for onset_frame in self.onset_frames[applicable_runs]])
        stimulus_durations = np.array([[stimulus_duration] for stimulus_duration in self.stimulus_durations[applicable_runs]])

        return onset_frames, stimulus_durations

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
        """
        Apply a boxcar filter to the provided data array.

        The boxcar filter smooths the input data using a moving average, with
        the averaging window defined by `filter_length`. The filter can be applied
        with respect to past, future, or current data points, as specified by the
        `time` parameter.

        Parameters:
        data (np.array): The input data array to be filtered.
        filter_length (int): The number of data points to include in the moving
                            average window. It should be a positive integer.
        sign (float): A multiplier for the filter window. Typically 1 or -1,
                    representing a standard or inverted filter.
        time (str, optional): Determines the alignment of the filter window.
                            Options are "past", "future", or "current".
                            Default is "current".

                            "past" - The window is aligned such that the filter
                                    considers only past data points.
                            "future" - The window is aligned to include only future
                                        data points.
                            "current" - The window is centered around each data
                                        point, considering both past and future
                                        data points.

        Returns:
        np.array: The filtered data array. This array has the same length as the
                input data, but with the specified filtering applied.

        Notes:
        - The function pads the input data with zeros at the start or end (or both)
        to maintain the same array length after filtering.
        - The division by `filter_length` normalizes the filter, ensuring that the
        magnitude of the data remains consistent.
        """

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

    def get_homing_speed(self) -> np.array:
        """Extraction of homing speed from tracking data returns max speed relative to any reference location per frame

        Overview:
        + Computing the speed relative to each reference location per frame
        + Reference locations = [shelter, subgoal1, subgoal2]

        Assumptions:
        +  It assumes that the most relevant movement of the mouse in each frame is the
        one where it moves fastest relative to any of the reference locations
        + Smoothes the speed with a gaussian filter

        Returns:
        + homing_speed: np.array of shape (n_frames, ) with the speed in cm/s)"""

        # Initialize speed relative to reference locations
        speed_relative_to_reference_locations = np.zeros((len(self.tracking_data["avg_loc"][:, 0]) - 1, len(self.reference_locations)))

        # For each reference location, compute the speed relative per frame
        for i, reference_location in enumerate(self.reference_locations):
            # Find euclidean distance between reference location and mouse location
            distance_from_reference_location = (
                (self.tracking_data["avg_loc"][:, 0] - reference_location[0]) ** 2
                + (self.tracking_data["avg_loc"][:, 1] - reference_location[1]) ** 2
            ) ** 0.5
            speed_pixel_per_frame = -np.diff(distance_from_reference_location)
            speed_relative_to_reference_locations[:, i] = speed_pixel_per_frame

        # Choosing max speed relative to any reference location per frame is debatable and a bit arbitrary
        homing_speed_pixel_per_frame = np.max(speed_relative_to_reference_locations, axis=1)
        homing_speed_cm_per_sec = homing_speed_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_homing_speed_cm_per_sec = gaussian_filter1d(homing_speed_cm_per_sec, sigma=self.session.video.fps / 2)

        # Add a zero needed to ensure len as np.diff returns len-1
        homing_speed = np.concatenate((np.zeros(1), smoothed_homing_speed_cm_per_sec))

        # Speed cant be move than 100cm (arbitrary) per second as that would be ridiculous
        assert np.max(homing_speed) < 100, "Homing speed is too high, check tracking data"

        return homing_speed

    def get_homing_angle(self) -> np.array:
        """Computes the angle relative to each reference location per frame and returns the min angle

        NOTE - These angles are already computed in video df, so this could be removed
        and replaced with logic that calls that df instead of recomputing it here.
        Leaving it for now to avoid breaking anything / refactor time sink.

        Assumption:
        -- the most relevant movement of the mouse in each frame is the one that is min

        Returns:
        -- homing_angle: np.array of shape (n_frames, ) with the angle in degrees"""

        # Initialize angle relative to reference locations
        angle_relative_to_reference_locations = np.zeros((len(self.tracking_data["avg_loc"][:, 0]), len(self.reference_locations)))

        for i, reference_location in enumerate(self.reference_locations):
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

        # Doesn't seem to be used anywhere so commenting out for now
        # self.shelter_angle = abs(angle_relative_to_reference_locations[:, 0])

        # Choice of min angle is debatable and a bit arbitrary
        homing_angle = np.min(abs(angle_relative_to_reference_locations), axis=1)
        return homing_angle

    def get_homing_speed_angular(self) -> np.array:
        """Converts the homing angle into angular speed in degrees per second

        Assumptions:
        -- guassian filter is used to smooth the angular speed

        Returns:
        -- homing_speed_angular: np.array of shape (n_frames, ) with the angular speed in degrees per second"""
        angular_speed_deg_per_frame = -np.diff(self.homing_angle)
        angular_speed_deg_per_sec = angular_speed_deg_per_frame * self.session.video.fps
        smoothed_angular_speed_deg_per_sec = gaussian_filter1d(angular_speed_deg_per_sec, sigma=self.session.video.fps / 10)
        homing_speed_angular = np.concatenate((np.zeros(1), smoothed_angular_speed_deg_per_sec))
        return homing_speed_angular

    def get_speed_along_y_axis(self) -> np.array:
        """Return and smooth the speed along the y axis

        Returns:
        -- speed_along_y_axis: np.array of shape (n_frames, ) with the speed in cm/s along the y axis"""
        speed_y_pixel_per_frame = np.diff(self.tracking_data["avg_loc"][:, 1], axis=0)
        speed_y_cm_per_sec = speed_y_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_speed_y_cm_per_sec = gaussian_filter1d(speed_y_cm_per_sec, sigma=self.session.video.fps / 10)
        speed_along_y_axis = np.concatenate((np.zeros(1), smoothed_speed_y_cm_per_sec))
        return speed_along_y_axis
