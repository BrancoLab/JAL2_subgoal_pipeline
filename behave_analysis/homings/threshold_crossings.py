"""Again no idea what this does."""

from dataclasses import dataclass

import dill as pickle
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter1d
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration

@dataclass(frozen=True)
class Threshold_crossings:
    onset_frames: object
    stimulus_durations: object

class get_Threshold_crossings:
    def __init__(self, settings, session):
        self.settings = settings
        self.session = session

        self.extract_variables()
        self.identify_threshold_crossings()
        self.get_onset_and_duration()
        self.remove_inapplicable_runs()

        self.session.threshold_crossing = Threshold_crossings(self.onset_frames, self.stimulus_durations)
        self.save_session()

# --------MAIN FUNCS-----------------------------------------------
    def extract_variables(self):
        open_tracking_data(self) # generates self.tracking_data
        self.get_speed_along_each_axis()

        height = self.session.video.registration_size[1]
        width  = self.session.video.registration_size[0]
        self.roi = np.zeros((height, width))
        if self.session.experiment in ['block edge vectors', 'block after 2nd edge vector', 'open field','no laser']:
            cv2.circle(self.roi, (int(width/2-250), int(height/2 - 50)), 200, 100, -1)
        if self.session.experiment == 'block pre edge vectors':
            cv2.drawContours(self.roi, [np.array([(0,0),(380, 0),(930, height-1),(0, height-1)], dtype=int)], 0, 100, -1)
        if self.session.experiment == 'block post edge vectors':
            cv2.drawContours(self.roi, [np.array([(0, height/2),(width-1, height/2),(width-1, height-1),(0, height-1)], dtype=int)], 0, 100, -1)


    def identify_threshold_crossings(self):
        self.in_roi = self.roi[self.tracking_data['avg_loc'][:, 1].astype(int), self.tracking_data['avg_loc'][:, 0].astype(int)]

    def get_onset_and_duration(self):
        self.onset_frames, self.stimulus_durations, _ = \
            get_onset_and_duration(self.in_roi, self.session, stim_type='threshold crossings', min_frames_between_trials=2, data_type='frames')
        
        self.stimulus_durations = np.ones_like(self.stimulus_durations) * self.settings.duration_after_crossing

    def remove_inapplicable_runs(self):
        moving_downward = self.speed_along_y_axis[self.onset_frames] > 5
        moving_leftward       = self.speed_along_x_axis[self.onset_frames]          < 0
        above_lower_limit_1   = self.tracking_data['avg_loc'][self.onset_frames, 1] < 430
        right_of_left_limit_1 = self.tracking_data['avg_loc'][self.onset_frames, 0] > 120
        below_upper_limit_2 = self.tracking_data['avg_loc'][self.onset_frames, 1] > 120
        above_lower_limit_2 = self.tracking_data['avg_loc'][self.onset_frames, 1] < 400
        starts_late_enough  = (self.onset_frames / self.session.video.fps / 60)   <  self.settings.max_time_within_session

        if self.session.experiment in ['block edge vectors', 'block after 2nd edge vector', 'open field','no laser']:
            applicable_runs = moving_downward * above_lower_limit_1 * right_of_left_limit_1 * starts_late_enough
        if self.session.experiment == 'block pre edge vectors':
            applicable_runs = moving_leftward * below_upper_limit_2 * above_lower_limit_2 * starts_late_enough
        if self.session.experiment == 'block post edge vectors':
            applicable_runs = starts_late_enough
        
        self.onset_frames       = np.array([[onset_frame] for onset_frame in self.onset_frames[applicable_runs]])
        self.stimulus_durations = np.array([stimulus_duration for stimulus_duration in self.stimulus_durations[applicable_runs]])

    def save_session(self):
        with open(self.session.metadata_file, "wb") as dill_file: pickle.dump(self.session, dill_file)

# --------DATA PROCESSING FUNCS---------------------------------------
    def get_speed_along_each_axis(self):
        speed_xy_pixel_per_frame     = np.diff(self.tracking_data['avg_loc'], axis=0)
        speed_xy_cm_per_sec          = speed_xy_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_speed_xy_cm_per_sec = gaussian_filter1d(speed_xy_cm_per_sec, sigma=self.session.video.fps/10, axis=0)
        self.speed_along_x_axis      = np.concatenate((np.zeros(1), smoothed_speed_xy_cm_per_sec[:, 0]))
        self.speed_along_y_axis      = np.concatenate((np.zeros(1), smoothed_speed_xy_cm_per_sec[:, 1]))


