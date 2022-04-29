"""_summary_

A script to return the onset of TTL pulses to align the behavioural data 
collected on the big rig with the efizz data collected on the efizz machine.

Returns an object class containing:
- ttl pulse onsets
- the raw ttl signal

TODO:
- Look through fede's code to see if we need any other logic
- Look at the efizz rig and run similar checks - duplicate logic
"""

#Custom libaries
from behave_analysis.process.session import Session
from behave_analysis.utils.load_bin_or_np import load_or_open

#OS libaries
import os
import numpy as np
from dataclasses import dataclass
from glob import glob
import dill as pickle
import pandas as pd

#Store file name here now for testing
imec_bin_file = "C:/Users/JoannaA/Desktop/data/ephys/test0_g0_imec0/test0_g0_t0.imec0.ap.bin"

@dataclass(frozen=False)
class TTL_Sync:
    # Storing relevant data to align big rig with efizz machine using the onset of TTL pulses
    bonsai_TTL: float # voltage recordings of ttl signal from bonsai machine
    pulse_index: int # pulse onset index from bonsai machine
    imec_TTL: float
    sampling_rate: int # Should be 30khz for neuropixels

def get_TTL(session: Session) -> TTL_Sync:
    """_summary_
    Returns the TTL_sync class containing the onset of TTL pulses.

    Args:
        session (Session): custom object containing experimental path file

    Returns:
        TTL_Sync: TTL_Sync.pulse_onset can be used to sync with another machine
    """
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    ttl_signal = AI_data[3:-1:4] #From the 3 index until the end select every 4th sample
    ttl_pulse_index = find_pulse_index(ttl_signal)
    imec_TTL = get_TTL_from_imec(imec_bin_file)
    ttl_object = TTL_Sync(ttl_signal, ttl_pulse_index, imec_TTL, 30000) #define final output
    return (ttl_object)

def get_TTL_from_imec(filename):
    data = load_or_open(filename, "int16", order="F", dtype="int16")
    return(data)

def find_pulse_index(ttl_signal):
    """_summary_
    A function that returns the index of pulse onset.

    Args:
        ttl_signal (_type_): A raw TTL signal voltage recording
    """
    ttl_pulses_diff = np.diff(ttl_signal) #Compute the difference between xi+1-xi for len array
    ttl_pulses_idx  = np.where(ttl_pulses_diff > 1)[0] + 1 #plus one as diff index shifts
    return(ttl_pulses_idx)