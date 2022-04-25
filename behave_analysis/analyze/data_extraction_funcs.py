from typing import Tuple
import numpy as np
from behave_analysis.analyze.analysis_funcs import *

def create_trial_dict(self, trial_start_idx: int, trial_end_idx: int) -> dict:
    trial = {}
    trial['mouse']                 = self.session.mouse
    trial['session count']         = self.session_count
    trial['trial count']           = self.trial_num
    trial['group number']          = self.group_num
    trial['experiment']            = self.session.experiment
    trial['trial start']           = trial_start_idx
    trial['speed']                 = self.tracking_data['speed'][trial_start_idx+1:trial_end_idx]
    trial['trajectory x']          = self.tracking_data['avg_loc'][trial_start_idx:trial_end_idx, 0]
    trial['trajectory y']          = self.tracking_data['avg_loc'][trial_start_idx:trial_end_idx, 1]
    trial['escape end idx']        = trial_end_idx
    trial['escape initiation idx'] = get_escape_initiation_idx(self, trial_start_idx)
    trial['escape target score']   = get_escape_target_score(self, trial['trajectory x'], trial['trajectory y'], trial['escape initiation idx'])
    trial['which side']            = get_which_side(self, trial_start_idx)

    if'trajectories' in self.analysis_type and self.settings.reflect_trajectories and get_which_side(self, trial_start_idx)=='right':
        trial['trajectory x']  = self.session.video.registration_size[0] - trial['trajectory x']
    
    if self.stim_type in ['homing', 'threshold_crossing'] and 'block' in self.session.experiment:
        trial['escape target score'] = get_escape_target_score(self, self.tracking_data['head_loc'][trial_start_idx:trial_end_idx, 0], \
                                                                     self.tracking_data['head_loc'][trial_start_idx:trial_end_idx, 1], \
                                                                     trial['escape initiation idx'])
        trial['frames before laser'] = min(abs(trial_start_idx - np.array([onsets[0] for onsets in self.session.laser.onset_frames])))

    if 'trial' in self.analysis_type:
        trial['head_dir'] = 180-self.tracking_data['head_dir'][trial_start_idx:trial_end_idx]
        trial['neck_dir'] = 180-self.tracking_data['neck_dir'][trial_start_idx:trial_end_idx]
        trial['body_dir'] = 180-self.tracking_data['body_dir'][trial_start_idx:trial_end_idx]
        trial['upper_body_dir'] = np.mean((trial['neck_dir'], trial['body_dir']), 0)
        trial['lower_body_dir'] = np.mean((trial['body_dir'], trial['upper_body_dir']), 0)
        trial['shoulder_dir']   = np.mean((trial['neck_dir'], trial['upper_body_dir']), 0)

        trial['head_loc']       = self.tracking_data['head_loc'][trial_start_idx:trial_end_idx,:]
        trial['body_loc']       = self.tracking_data['body_loc'][trial_start_idx:trial_end_idx,:]
        trial['upper_body_loc'] = self.tracking_data['upper_body_loc'][trial_start_idx:trial_end_idx, :]
        trial['lower_body_loc'] = np.mean((trial['body_loc'], trial['upper_body_loc']), axis=0)
        trial['neck_loc']       = np.mean((trial['head_loc'], trial['upper_body_loc']), axis=0)
        trial['shoulder_loc']   = np.mean((trial['neck_loc'], trial['upper_body_loc']), axis=0)

    return trial

def get_trial_start_and_end(self, onset_frames: list) -> Tuple[int, int]:
    trial_start_idx = onset_frames[0]
    trial_end_idx = trial_start_idx + min(12*self.fps, get_to_shelter_idx(self, trial_start_idx))
    print(trial_end_idx - trial_start_idx)
    return trial_start_idx, trial_end_idx