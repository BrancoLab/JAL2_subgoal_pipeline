"""A script to return the onset of TTL pulses for both the imec and bonsai machine.
This is to align the behavioural data collected on the big rig with the efizz data collected on 
the imec machine.

Returns an object class containing:
- bonsai_TTL: The TTL from the big rig machine
- imec_TTL: The TTL from the imec bin file from the efizz machine
- sampling_rate: This is hard coded at 30khz
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
import random

#Store file name here now for testing - hard coded need to update
imec_bin_file = "C:/Users/JoannaA/Desktop/data/ephys/test0_g0_imec0/test0_g0_t0.imec0.ap.bin"

@dataclass(frozen=False)
class TTL_Sync:
    # Storing relevant data to align big rig with efizz machine using the onset of TTL pulses
    bonsai_TTL: float # voltage recordings of ttl signal from bonsai machine
    imec_TTL: float
    sampling_rate: int # Should be 30khz for neuropixels

#Return the above data class
def get_TTL(session: Session) -> TTL_Sync:
    """Returns the TTL_sync dataclass. 

    Args:
        session (Session): custom object containing experimental path file

    Returns:
        TTL_Sync: data class
    """
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    bonsai_ttl = AI_data[3:-1:4] #From the 3 index until the end select every 4th sample
    imec_TTL = get_TTL_from_imec(imec_bin_file)
    ttl_object = TTL_Sync(bonsai_ttl, imec_TTL, 30000) #define final output
    return (ttl_object)

#Load imec bin file
def get_TTL_from_imec(filename):
    """Load the imec bin file and convert to np memory map

    Args:
        filename (str): File name of .bin imec file produced by spikeGLX
    """
    data = load_or_open(filename, "int16", order="F", dtype="int16")
    return(data)

#If the signal pulses are not of the same duration, this function is used
def remove_idx_to_align_signals(bonsai_onsets, bonsai_signal, temporal_diff):
    """A function that  removes samples from each pulse for the bonsai signal to ensure the
    duration of each pulse is the same as the efizz signal. This will then
    allow another function to shift the signals so they can be aligned. This assumes
    the bonsai machine is slower than the imec machine.

    Args:
        bonsai_onsets (_type_): index of pulse onset for bonsai pulse
        bonsai_signal (_type_): TTL signal out of bonsai machine
        temporal_diff (_type_): temporal_difference = delta_bonsai_onsets - delta_ephys_onsets # The difference in pulse lengths

    Returns:
        np array: A updated bonsai TTL signal that should have the same pulse intervals as
        the matching efizz rig

    To do:
    - # off by one error need to fix
    - # make faster it takes like 10 minutes
    """
    #Copy signal to conduct length test
    copy_of_original_signal_for_test = np.copy(bonsai_signal)

    #For each pulse, remove n samples uniformly 
    for pulse in range(len(bonsai_onsets) - 1):
        first_pulse_idx = bonsai_onsets[pulse] # index of pulse
        next_pulse_idx  = bonsai_onsets[pulse + 1] # index of next pulse
        num_samples_to_remove = temporal_diff[pulse] # how many samples to remove this pulse
        print(pulse) # print current pulse as func is slow so need to check where at
        for sample in range(num_samples_to_remove):
            choose_index = random.randint(first_pulse_idx, next_pulse_idx) # generate random sample to remove between onsets
            bonsai_signal = np.delete(bonsai_signal, choose_index) # delete that index
    assert (len(bonsai_signal) == len(copy_of_original_signal_for_test) - sum(temporal_diff)), "The new signal does not match the old - number of changes required" 
    return (bonsai_signal)
