# Custom libs
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration
from behave_analysis.utils.AI_dataClass_objects import Audio

# OS Libs
import os
import numpy as np
from glob import glob
import dill as pickle

def get_Audio(session: NEW_Session) -> Audio:
    """AI data is a 4 channel interleaved signal. The audio signal is the second channel.
    AI stands for analog input. The audio signal is an offshot of the signal sent to the speaker and 
    equals a voltage recording. 

    Args:
        session (Session): _description_
        indexs_to_remove (_type_, optional): _description_. Defaults to None.
        down_sample (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
        
    AI_file = list(session.file_path.glob("*analog.bin"))[0] # need lst and idx as its a generator

    if '.bin' in str(AI_file): 
        AI_data = np.fromfile(AI_file)
        
    else: 
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)        
    
    audio_data = AI_data[np.arange(1, len(AI_data), 4)] # four interleaved time series
    audio_num_samples = len(audio_data)
    audio_on = abs(audio_data)>3
    audio_onset_frames, stimulus_durations, _ = get_onset_and_duration(audio_on, 
                                                                       session, 
                                                                       stim_type='audio', 
                                                                       min_frames_between_trials = session.daq_sampling_rate * 5, 
                                                                       data_type='samples')
    
    audio = Audio(audio_num_samples, audio_onset_frames, stimulus_durations)
    return audio