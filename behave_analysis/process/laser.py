from behave_analysis.process.session import Session
import os
import numpy as np
import itertools
from dataclasses import dataclass
from typing import Tuple
from glob import glob
import dill as pickle

@dataclass(frozen=True)
class Laser:
    num_samples: int
    onset_frames: object
    stimulus_durations: object
    frequency: object

def get_Laser(session: Session) -> Laser:
    if 'initiation' in session.experiment: return None
    laser_file = glob(os.path.join(session.file_path, "laser*"))[-1] # take the last file if there are multiple
    if '.bin' in laser_file: 
        laser_data = np.fromfile(laser_file)
    else: 
        with open(laser_file, "rb") as dill_file: laser_data = pickle.load(dill_file)
    num_samples = len(laser_data)
    onset_frames, stimulus_durations, frequency = get_laser_stimulus_parameters(laser_data, session)
    laser = Laser(num_samples, onset_frames, stimulus_durations, frequency)
    return laser

def get_laser_stimulus_parameters(laser_data: object, session: Session) -> Tuple[object, object, object]:
    # GETS THE FRAME NUMBERS WHEN THE LASER TURNS ON (AUTOMATICALLY OR WAS CLICKED)
    laser_trial_durations = []
    laser_trial_onset_idx = []
    laser_trial_frequency = []
    laser_data_diff = np.diff(laser_data)
    laser_pulse_onset_idx = np.where(laser_data_diff > .001)[0] + 4001 #! specific to particular bonsai implementation
    if not laser_pulse_onset_idx.size:
        print("No laser trials detected")
        return [],[],[]
    laser_samples_since_last_start = np.diff(laser_pulse_onset_idx)
    current_pulse_idx = 0

    for cur_laser_num_consecutive_samples, group_object in itertools.groupby(laser_samples_since_last_start):
        num_laser_onsets_in_this_group = len(list(group_object))
        if cur_laser_num_consecutive_samples < 2000:
            laser_trial_durations.append(np.round(
                cur_laser_num_consecutive_samples * num_laser_onsets_in_this_group / session.daq_sampling_rate))
            laser_trial_onset_idx.append(
                laser_pulse_onset_idx[current_pulse_idx])
            laser_trial_frequency.append(
                np.round(session.daq_sampling_rate / cur_laser_num_consecutive_samples))
        current_pulse_idx += num_laser_onsets_in_this_group

    laser_onset_frames = np.array([np.argmin(abs(x - session.camera_trigger.frame_trigger_onsets_idx)) for x in laser_trial_onset_idx])
    frames_since_previous_laser_onset = np.append(99999, np.diff(laser_onset_frames))
    start_of_stimulus_train = frames_since_previous_laser_onset > (10 * session.camera_trigger.fps) # was 20
    stimulus_train_idx = np.array(
        [sum(start_of_stimulus_train[:i+1])-1 for i, s in enumerate(start_of_stimulus_train)])
    laser_onset_frames_grouped_by_stimulus_train = np.array(
        [laser_onset_frames[stimulus_train_idx == i] for i in range(stimulus_train_idx[-1]+1)], dtype=object)
    laser_duration_grouped_by_stimulus_train = np.array([np.array(laser_trial_durations)[stimulus_train_idx == i] for i in range(stimulus_train_idx[-1]+1)], dtype=object)
    laser_frequency = np.array(laser_trial_frequency)[start_of_stimulus_train]
    return laser_onset_frames_grouped_by_stimulus_train, laser_duration_grouped_by_stimulus_train, laser_frequency
