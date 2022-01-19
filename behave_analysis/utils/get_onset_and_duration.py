import numpy as np
from typing import Tuple

def get_onset_and_duration(data_on: object, session: object, stim_type: str, min_frames_between_trials: int, data_type: str) -> Tuple[object, object, object]:
    
    data_on_idx = np.where(data_on)[0]

    if not data_on_idx.size:
        print("No {} trials detected".format(stim_type))
        return [],[]

    idx_since_data_on     = np.append(np.inf, np.diff(data_on_idx)) 
    data_onset_idx        = data_on_idx[idx_since_data_on > min_frames_between_trials]
    
    idx_before_next_trial = np.append(-np.inf, np.diff(data_on_idx[::-1]))[::-1]
    data_offset_idx       = data_on_idx[idx_before_next_trial < -min_frames_between_trials]

    if data_type == 'samples':
        data_onset_frames     = np.array([[np.argmin(abs(x - session.camera_trigger.frame_trigger_onsets_idx))] for x in data_onset_idx])
        data_offset_frames    = np.array([[np.argmin(abs(x - session.camera_trigger.frame_trigger_onsets_idx))] for x in data_offset_idx])
    if data_type == 'frames':
        data_onset_frames  = data_onset_idx
        data_offset_frames = data_offset_idx

    if data_type=='samples':
        stimulus_durations = np.array([[x] for x in np.round((data_offset_idx-data_onset_idx) / session.daq_sampling_rate, 1)])
    if data_type=='frames': 
        stimulus_durations = np.array([[x] for x in (data_offset_idx-data_onset_idx) / session.video.fps])

    return data_onset_frames, stimulus_durations, data_offset_frames