"""A script to return the onset of TTL pulses for both the imec and bonsai machine.
This is to align the behavioural data collected on the big rig with the efizz data collected on 
the imec machine.

---------------------------------------------

TTL Checks contained within this script:
1. Are the TTL signals close in len by 10 seconds?
2. Are the pulses too short or too long?
3. Are there any spikes in the TTL signals above expected voltage thresholds?
4. Check if the signals have different pulse lenghts?

There are additional checks living in process.py that could be refactored into this script.

-----------------------------------------------

Note - the onsets/offsets are before the resample and thus if you plot them next to resample signal they will be missaligned

Returns an object class containing:
- bonsai_TTL: The TTL from the big rig machine
- imec_TTL: The TTL from the imec bin file from the efizz machine
- sampling_rate: This is hard coded at 30khz

Todo:
- update script summary with new class attributes
"""

#Custom libaries
from behave_analysis.process.session import Session
from behave_analysis.utils.load_bin_or_np import load_or_open
from databank import efizz

#OS libaries
import os
import numpy as np
from dataclasses import dataclass
from glob import glob
import dill as pickle
import pandas as pd
import random
from loguru import logger
import matplotlib.pyplot as plt

#Store file name here now for testing - hard coded need to update
imec_bin_file = efizz[0]

@dataclass(frozen=False)
class TTL_Sync:
    # Storing relevant data to align big rig with efizz machine using the onset of TTL pulses
    bonsai_TTL: float # voltage recordings of ttl signal from bonsai machine
    imec_TTL: float
    inital_bonsai_len: int # What is the length of the bonsai TTL before cleaning
    sampling_rate: int # Should be 30khz for neuropixels
    bonsai_sync_onsets: int # array of ints, onset/offsets PRE RESAMPLING 
    bonsai_sync_offsets: int # array of ints, onset/offsets PRE RESAMPLING 
    ephys_sync_onsets: int # array of ints, onset/offsets PRE RESAMPLING 
    ephys_sync_offset: int # array of ints, onset/offsets PRE RESAMPLING 
    temporal_difference: int # array of ints, differences in offsets and onsets
    choose_index: int # which indexs to delete, array of ints
    bonsai_obj: object # signal at differnet stages

#Return the above data class
def get_TTL(session: Session, down_sample = True) -> TTL_Sync:
    """Returns the TTL_sync dataclass. 

    Args:
        session (Session): custom object containing experimental path file

    Returns:
        TTL_Sync: data class
    """

    #Set global sampling rate
    sampling_rate = 30000

    # bonsai_obj
    bonsai_obj = {}

    # Retrieve TTL data
    bonsai_ttl, imec_TTL = retrieve_TTL_signals(session)
    logger.info("The length of the original bonsai TTL is: {} and the original imec TTL is: {}".format(len(bonsai_ttl), len(imec_TTL)))
    inital_bonsai_len = len(bonsai_ttl)
    if len(bonsai_ttl) > len(imec_TTL): logger.error("Bonsai TTL is longer than imec TTL this can't be")

    #Check and correct for abberant signals
    imec_TTL, bonsai_ttl = check_for_abberant_signals(bonsai_ttl, imec_TTL, sampling_rate)

    #Get onset and offsets - PRE DOWNSAMPLE
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(bonsai_ttl, 2.5)
    ephys_sync_onsets, ephys_sync_offsets   = get_onset_offset(imec_TTL, 45)

    # Check pulse lengths
    check_for_abberant_pulses(bonsai_sync_onsets, ephys_sync_onsets, sampling_rate)

    #Now compare delta between onsets and offsets
    delta_ephys_onsets  = np.diff(ephys_sync_onsets)
    delta_bonsai_onsets = np.diff(bonsai_sync_onsets)
    delta_ephys_offsets = np.diff(ephys_sync_offsets)
    delta_bonsai_offsets = np.diff(bonsai_sync_offsets)

    #If down_sample is set to true and there exsists a delta between onsets of efiz and bonsai, resample bonsai signal
    if down_sample and not ((delta_ephys_onsets==delta_bonsai_onsets).all() or (delta_ephys_offsets==delta_bonsai_offsets).all()):
        #Check that the interval pulses match between imec and bonsai
        #Difference in temporal scale
        temporal_difference = delta_bonsai_onsets - delta_ephys_onsets # Compare the difference in pulse lengths
        r_all_bonsai_pulses_longer = temporal_difference >= 0
        is_it_true = r_all_bonsai_pulses_longer.all()
        if not is_it_true: logger.error("Bonsai pulse onsets are not all greater than or equal to imec pulse onsets")
        logger.warning("Pulse lengths do not match between imec and efizz. The signals require modifcation before shifting")
        if (sum(temporal_difference) > 0):
            logger.warning("Bonsai pulses are longer than efizz pulses. Resampling signal.")
            bonsai_ttl, choose_index = remove_idx_to_align_signals(bonsai_sync_onsets, bonsai_ttl, temporal_difference)
        else: 
            logger.error("Bonsai recording is shorter than the ephys recording, ending script as this goes against assumptions")
            return

    # Align with the first onset and remove start of signal
    bonsai_obj['post resample'] = bonsai_ttl
    bonsai_ttl, imec_TTL = remove_start_of_signal(imec_TTL, 
                                                  ephys_sync_onsets,
                                                  bonsai_ttl,
                                                  bonsai_sync_onsets)

    logger.info("The length of the Imec signal from the first pulse until the end is: {}".format(len(imec_TTL)))

    # compute onset off setts after resample and alignment
    bonsai_obj['post start of signal cut'] = bonsai_ttl
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(bonsai_ttl, 2.5)
    ephys_sync_onsets, ephys_sync_offsets   = get_onset_offset(imec_TTL, 45)

    # remove the flat end of the TTL signals
    imec_TTL, bonsai_ttl = remove_end_of_TTLs(imec_TTL, 
                                              ephys_sync_offsets,
                                              bonsai_ttl,
                                              bonsai_sync_offsets)


    # define the TTL object
    ttl_object = TTL_Sync(bonsai_ttl, 
                          imec_TTL,
                          inital_bonsai_len, 
                          sampling_rate,
                          bonsai_sync_onsets,
                          bonsai_sync_offsets,
                          ephys_sync_onsets,
                          ephys_sync_offsets,
                          temporal_difference,
                          choose_index,
                          bonsai_obj)
    return (ttl_object)

#--------------------------------------- Load and retrieve data functions -----------------------------------------------------------------------

# Retrieve TTL signals
def retrieve_TTL_signals(session: Session):
    """Retrieves TTL signals for both the bonsai machine and the imec machine

    Args:
        session (Session): _description_

    Returns:
    - TTL signals from bonsai machine and imec board
    """
    #Retrieve TTL data
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    bonsai_ttl = AI_data[np.arange(3, len(AI_data), 4)] #From the 4 index until the end select every 4th sample
    imec_TTL = get_TTL_from_imec(imec_bin_file)
    return bonsai_ttl, imec_TTL

#Load imec bin file
def get_TTL_from_imec(filename):
    """Load the imec bin file and convert to np memory map

    Args:
        filename (str): File name of .bin imec file produced by spikeGLX
    """
    data = load_or_open(filename, "int16", order="F", dtype="int16")
    return(data)

# ---------------------------------------------- utils ------------------------------------------------------------------

#Get the onsets and offsets for bonsai / imec. Your choice!
def get_onset_offset(signal, threshold, clean=True):
    """ Get onset/offset times when a signal goes below>above and
        above>below a given threshold

        Arguments:
            signal: 1d numpy array
            thhreshold: float, threshold
            clean: bool. If true ends before the first start and 
                starts after the last end are removed

        Returns:
            Starts: Indexes of pulse onsets
            Ends: Indexes of pulse offsets
    """
    above = np.zeros_like(signal) # Creates an array of zeros of length signal
    above[signal >= threshold] = 1 #If the signal is above voltage threshold, set to 1
    der = derivative(above) #Create an array of differences 
    starts = np.where(der > 0)[0] #Where does the signal switch from 0 to 1
    ends = np.where(der < 0)[0] #Where does the signal switch from 1 to 0

    #If the signal starts with a pulse add a zero to the start
    if above[0] > 0:
        starts = np.concatenate([[0], starts])

    #If the signal ends at the top of the pulse add the length of the signal to the end
    if above[-1] > 0:
        ends = np.concatenate([ends, [len(signal)]])

    #If clean is true
    if clean:
        # ends before the first start are removed
        ends = np.array([e for e in ends if e > starts[0]])

        # starts before the last end are removed
        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    #If there aren't any starts or ends create empty arrays
    if not np.any(starts):
        starts = np.array([0])
        logger.error("No onsets")
    if not np.any(ends):
        ends = np.array([len(signal)])
        logger.error("No offsets")
    return starts, ends

#Take the derivate so you can spot changes in state within a signal. Ie, pulse onset/offsets
def derivative(X, axis=0, order=1):
    """"Takes the derivative of an array X along a given axis

        Arguments:
            X: np.array with data - 1 dimensional
            axis: int. Axis along which the derivative is to be computed
            order: int. Derivative order
    """
    #Prepend 0 so the index is realigned to prevent off by 1 error
    return np.diff(X, n=order, axis=axis)

#-------------------------------------- Abberant signal checks --------------------------------------------------------------------------

#Check for abberant signals and errors
def check_for_abberant_signals(bonsai_ttl, imec_TTL, sampling_rate):
    """A function that checks:
    1) If the signal lengths are too different between bonsai and immec, they shouldn't be longer than
    10 seconds. Enabling bonsai and disabling bonsai and spike glx shouldn't take >10s but if mannually this occured
    then this error would fire, or if another issue occured
    2) If the signals deviate away from an expected baseline

    Args:
        bonsai_ttl (object): TTL signal
        imec_TTL (object): TTL signal
        sampling_rate (int): FS of sampling system

    Returns:
        Object: Imec TTL Signal cleaned from abberant spikes
        Object: bonsai TTL signal cleaned from abberant spikes
    """

    # Threshold for acceptable number of abberant signals made up
    threshold = 1000

    # check for signal differences, they should not differ by 10 seconds
    if abs(len(bonsai_ttl) - len(imec_TTL)) > 10 * sampling_rate:
        logger.warning("The sync signals have very different lengths before resample, this cant be!")
        return

    # check for aberrant signals in ephys
    errors = np.where(imec_TTL > 75)[0]
    if len(errors):
        logger.warning(f"Found {len(errors)} samples with too high values in probe signal")
    if len(errors) > 1000:
        return False, 0, 0, "too_many_errors_in_ephys_sync_signal"

    # check of abberaant signals in bonsai TTL
    errors_bonsai = np.where(bonsai_ttl > 10)[0]
    assert len(errors_bonsai) < threshold, "There are too many abberant signals in the bonsai TTL"

    # If errors remove signals and update signals
    imec_TTL = np.delete(imec_TTL, errors)
    bonsai_ttl = np.delete(bonsai_ttl, errors_bonsai)
    logger.warning("Removing {} abberant signals from imec and {} from bonsai".format(len(errors), len(errors_bonsai)))
    
    # Log success
    logger.info("Bonsai and Imec TTL are of similar lengths and have passed the abberant signal verification ")

    return imec_TTL, bonsai_ttl

#Highlight weird length pulses
def check_for_abberant_pulses(bonsai_sync_onsets, ephys_sync_onsets, sampling_rate):
    """A function that checks if the delta between onsets and offsets is not greater or less than 
    what should roughly be expected. Are pulse lengths to be expected?

    Args:
        bonsai_sync_onsets (_type_): _description_
        ephys_sync_onsets (_type_): _description_
        sampling_rate (_type_): _description_
    """

    logger.info("Checking if pulse lenghts are as expected")

    #Bonsai pulse length check if too long or short
    bonsai_pulse_len_under_errors = np.where(np.diff(bonsai_sync_onsets) < (sampling_rate / 2))[0]
    bonsai_pulse_len_over_errors = np.where(np.diff(bonsai_sync_onsets) > (sampling_rate * 1.5))[0]
    if bonsai_pulse_len_under_errors or bonsai_pulse_len_over_errors:
        logger.error("Bonsai pulse greater or less than expected 1hz duration")
        return

    #Imec pulse length check if too long or short
    imec_pulse_len_under_errors = np.where(np.diff(ephys_sync_onsets) < (sampling_rate / 2))[0]
    imec_pulse_len_over_errors  = np.where(np.diff(ephys_sync_onsets) > (sampling_rate * 1.5))[0]
    if imec_pulse_len_under_errors or imec_pulse_len_over_errors:
        logger.error("Imec pulse greater or less than expected 1hz duration")
        return

#----------------------------------------------- Clean signals -------------------------------------------------------------------------------------------

#If the signal pulses are not of the same duration, this function is used
def remove_idx_to_align_signals(bonsai_onsets, bonsai_signal, temporal_diff):
    """A function that  removes samples from each pulse for the bonsai signal to ensure the
    duration of each pulse is the same as the efizz signal. This will then
    allow another function to shift the signals so they can be aligned. This assumes
    the bonsai machine is slower than the imec machine.

    Args:
        bonsai_onsets (array): index of pulse onset for bonsai pulse
        bonsai_signal (array): TTL signal out of bonsai machine
        temporal_diff (array): temporal_difference = delta_bonsai_onsets - delta_ephys_onsets # The difference in pulse lengths

    Returns:
        np array: 
        + A updated bonsai TTL signal that should have the same pulse intervals as
        the matching efizz rig
        + the indexs removed to be used for downsampling the other AI data

    """
    #Copy signal to conduct length test
    copy_of_original_signal_for_test = np.copy(bonsai_signal)

    #Create an array for the indexs to remove
    choose_index = np.array([])

    #For each pulse, remove n samples uniformly 
    for pulse in range(len(bonsai_onsets) - 1):
        #Take the number of samples needed to remove. Add one and don't select it. To ensure uniformity.
        choose_index = np.append(choose_index, np.linspace(bonsai_onsets[pulse] + 1, bonsai_onsets[pulse + 1], temporal_diff[pulse] + 1, dtype='int')[:-1])
    choose_index = list(choose_index.astype(int))
    bonsai_signal = np.delete(bonsai_signal, choose_index) # delete that index - all at once

    #Tests
    assert len(choose_index) == sum(temporal_diff), "The number of indexes choosen should equal the amount of samples required for removal"
    assert (len(bonsai_signal) == len(copy_of_original_signal_for_test) - sum(temporal_diff)), "The new signal does not match the old - number of changes required"
    assert (temporal_diff[3:] < 3000).all(), "Pulse onset difference between signals is greater than one milisecond. This is a considerable difference. Check pulses"
    # assert all(derivative(choose_index) > 1000), "A re_sample is less than 1000 samples apart. Not uniform "

    #Logs
    logger.info("Original bonsai signal length: {}, Num indexs to remove: {}, New signal length: {}".format(len(copy_of_original_signal_for_test), len(choose_index), len(bonsai_signal))) #continue to figure out off by one error

    return (bonsai_signal, choose_index)

# Remove the flat line end at the end of the imec signal
def remove_end_of_TTLs(imec_TTL, 
                       ephys_sync_offsets,
                       bonsai_TTL,
                       bonsai_sync_offsets):
    """Remove the end flat line of the imec signal

    Args:
        imec_TTL (_type_): _description_
        ephys_sync_offsets (_type_): _description_
    """
    logger.info("Removing the end of TTLs signal, cutting of at last offset")
    imec_TTL   = imec_TTL[:ephys_sync_offsets[-1]]
    bonsai_TTL = bonsai_TTL[:bonsai_sync_offsets[-1]]
    return imec_TTL, bonsai_TTL

# remove the signal prior to the first onset
def remove_start_of_signal(imec_TTL, 
                           ephys_sync_onsets,
                           bonsai_TTL,
                           bonsai_sync_onsets):
    
    # Get first index
    bonsai_first_pulse_idx = bonsai_sync_onsets[0]
    ephys_first_pulse_idx  = ephys_sync_onsets[0]

    #Index from the first pulse onset until the end of the signal and return the aligned signals
    bonsai_TTL_aligned = bonsai_TTL[bonsai_first_pulse_idx:]
    imec_TTL_aligned   = imec_TTL[ephys_first_pulse_idx:]

    return bonsai_TTL_aligned, imec_TTL_aligned
