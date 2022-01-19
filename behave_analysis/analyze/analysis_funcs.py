import numpy as np
from typing import Tuple


def get_escape_initiation_idx(self, trial_start_idx: int) -> int:
    if self.stim_type == 'laser':  return None
    if self.stim_type in ['homing', 'threshold_crossing']: return 0
    escape_initiation_idx = np.where(self.tracking_data['speed rel. to shelter'][trial_start_idx+1:] > self.settings.escape_initiation_speed)[0][0]
    assert escape_initiation_idx < (self.settings.max_escape_duration*self.fps)
    return escape_initiation_idx

def get_to_shelter_idx(self, trial_start_idx: int) -> int:
    if self.stim_type in ['homing', 'threshold_crossing', 'laser']:  return None
    to_shelter_idx = np.where(self.tracking_data['avg_loc'][trial_start_idx:, 1] > 850)[0][0]
    return to_shelter_idx

def get_escape_target_score(self, x: np.ndarray, y: np.ndarray, RT: int) -> float:
    if self.stim_type=='laser':  return None
    if self.stim_type in ['homing', 'threshold_crossing']: return gets_to_edge(self, x, y)

    x_start, y_start, x_goal, y_goal, _, _, x_target, y_target, x_obstacle_edge, y_obstacle_edge, _ = get_various_x_and_y_locations(self, x, y, RT)

    distance_path_to_homing_vector, _           = distance_to_line(x_target, y_target, x_start, y_start, x_goal, y_goal)
    distance_path_to_edge_vector, x_edge_vector = distance_to_line(x_target, y_target, x_start, y_start, x_obstacle_edge, y_obstacle_edge)
    distance_edge_vector_to_homing_vector, _    = distance_to_line(x_edge_vector, y_target, x_start, y_start, x_goal, y_goal)

    escape_target_score = abs(distance_path_to_homing_vector - distance_path_to_edge_vector + distance_edge_vector_to_homing_vector) / \
                            (2*distance_edge_vector_to_homing_vector)
    return escape_target_score

def distance_to_line(x, y, x_start, y_start, x_goal, y_goal) -> Tuple[float, float]:
    slope = (y_goal - y_start) / (x_goal - x_start)
    intercept = y_start - x_start * slope
    distance_of_path_to_line = abs(y - slope * x - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
    x_pos_at_y_value = (y - intercept) / slope
    return distance_of_path_to_line, x_pos_at_y_value
 
def get_various_x_and_y_locations(self, x: np.ndarray, y: np.ndarray, RT: int) -> tuple:
    y_start  = y[RT]
    x_start  = x[RT]
    y_goal   = self.session.video.shelter_location[1]
    x_goal   = self.session.video.shelter_location[0]
    y_center = self.session.video.registration_size[1]/2
    x_center = x[np.argmin(abs(y-y_center))]
    y_target         = y_center - 100 # 10cm in front of the wall
    x_target         = x[np.argmin(abs(y-y_target))]
    y_obstacle_edge  = y_center
    _, x_homing_vector = distance_to_line(x_target, y_target, x_start, y_start, x_goal, y_goal)
    x_obstacle_edge = self.session.video.registration_size[0]/2 + np.sign(x_target - x_homing_vector) * 250
    return x_start, y_start, x_goal, y_goal, x_center, y_center, x_target, y_target, x_obstacle_edge, y_obstacle_edge, x_homing_vector

def gets_to_edge(self, x: np.ndarray, y: np.ndarray) -> int:
    x_values_to_examine = x[y>self.session.video.registration_size[1]/2-50]
    x_dist_from_center  = abs(x_values_to_examine - self.session.video.registration_size[0]/2)
    if (x_dist_from_center>250).any(): return 1
    else: return 0

def get_which_side(self, trial_start_idx: int) -> str:
    if self.stim_type in ['homing', 'threshold_crossing'] and 'block' in self.session.experiment:
        frames_before_laser = min(abs(trial_start_idx - np.array([onsets[0] for onsets in self.session.laser.onset_frames])))
        if frames_before_laser < 2 * self.session.video.fps: return 'left'

    y_center = self.session.video.registration_size[1]/2 - 50 * (self.stim_type=='homing')
    cross_center_idx = np.where(self.tracking_data['avg_loc'][trial_start_idx:, 1] >= y_center)[0][0]
    x_at_cross = self.tracking_data['avg_loc'][trial_start_idx+cross_center_idx, 0]
    if x_at_cross < self.session.video.registration_size[0]/2: which_side = 'left'
    if x_at_cross > self.session.video.registration_size[0]/2: which_side = 'right'
    return which_side