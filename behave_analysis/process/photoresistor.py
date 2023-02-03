#Custom libs
from settings.settings_process import settings_process as settings_p # So to check if pipeline includes efizz

# Additional libraries if running with efizz
if settings_p.efizz:
    from behave_analysis.utils.downsample_AI_data import remove_idx_as_per_bonsai_ttl_resample

from turtle import down
from behave_analysis.process.session import Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration

# Os libs
import os
import numpy as np
from dataclasses import dataclass
from glob import glob
import dill as pickle
from loguru import logger

@dataclass(frozen=True)
class photoresistor_trigger:
    num_samples: int
    onset_frames: object
    stimulus_durations: object

def get_Photoresistor(session: Session, indexs_to_remove = None, down_sample = True) -> photoresistor_trigger:
    """AI data is a 4 channel interleaved signal. The photoresistor voltage is the third signal.
    AI stands for analog input. This is the voltage recording of the photoresistor. 
    """
    
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
            AI_data = np.fromfile(AI_file)
    else: 
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file) 
    resistor_data = AI_data[np.arange(2, len(AI_data), 4)] # four interleaved time series
    
    if down_sample:
        resistor_data = remove_idx_as_per_bonsai_ttl_resample("photo resist", 
                                                               resistor_data, 
                                                               indexs_to_remove, 
                                                               session)
    num_samples = len(resistor_data)
    resistor_on = resistor_data < 4.8
    resistor_onset_frames, stimulus_durations, _ = get_onset_and_duration(resistor_on, session, stim_type='resistor', min_frames_between_trials=session.daq_sampling_rate * 30, data_type='samples')
    photoresistor = photoresistor_trigger(num_samples, resistor_onset_frames, stimulus_durations)
    return (photoresistor)