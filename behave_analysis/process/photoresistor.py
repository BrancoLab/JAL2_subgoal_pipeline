#Custom libs
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration
from behave_analysis.utils.AI_dataClass_objects import photoresistor_trigger

# Os libs
import os
import numpy as np
from glob import glob
import dill as pickle

def get_Photoresistor(session: NEW_Session) -> photoresistor_trigger:
    """AI data is a 4 channel interleaved signal. The photoresistor voltage is the third signal.
    AI stands for analog input. This is the voltage recording of the photoresistor. """
    
    AI_file = list(session.file_path.glob("*analog.bin"))[0] # need lst and idx as its a generator

    if '.bin' in str(AI_file): 
            AI_data = np.fromfile(AI_file)
    else: 
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file) 
        
    resistor_data = AI_data[np.arange(2, len(AI_data), 4)] # four interleaved time series
    
    num_samples = len(resistor_data)
    resistor_on = resistor_data < 4.8
    resistor_onset_frames, stimulus_durations, _ = get_onset_and_duration(resistor_on, session, stim_type='resistor', min_frames_between_trials=session.daq_sampling_rate * 30, data_type='samples')
    photoresistor = photoresistor_trigger(num_samples, resistor_onset_frames, stimulus_durations)
    return (photoresistor)