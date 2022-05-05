#Custom libs
from behave_analysis.process.session import Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration
from behave_analysis.utils.downsample_AI_data import remove_idx_as_per_bonsai_ttl_resample

#OS Libs
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from glob import glob
import dill as pickle

@dataclass(frozen=True)
class Audio:
    num_samples: int
    onset_frames: object
    stimulus_durations: object

def get_Audio(session: Session, indexs_to_remove, temporal_diff, down_sample = True) -> Audio:
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else: 
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)        
    audio_data = AI_data[np.arange(1, len(AI_data), 4)] # four interleaved time series
    if down_sample: 
        audio_data = remove_idx_as_per_bonsai_ttl_resample("audio", audio_data, indexs_to_remove, temporal_diff)
    audio_num_samples = len(audio_data)
    audio_on = abs(audio_data)>3
    audio_onset_frames, stimulus_durations, _ = get_onset_and_duration(audio_on, session, stim_type='audio', min_frames_between_trials=session.daq_sampling_rate * 30, data_type='samples')
    audio = Audio(audio_num_samples, audio_onset_frames, stimulus_durations)
    return audio