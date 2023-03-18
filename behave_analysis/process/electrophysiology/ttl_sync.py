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

TODO:
- update script summary with new class attributes
- Add a check to see that the first pulse is larger than the rest, though not currently implemented in bonsai
"""

#Custom libaries
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.load_bin_or_np import load_or_open
from behave_analysis.utils.AI_dataClass_objects import TTL_Sync

#OS libaries
import os
import numpy as np
from glob import glob
import dill as pickle
from loguru import logger


def get_TTL(session: NEW_Session, TTL_bin_path: str):
    """Returns the TTL_sync dataclass. 

    Args:
        session (Session): custom object containing experimental path file

    Returns:
        TTL_Sync: data class
    """

    #Set global sampling rate
    sampling_rate = 30000

    # Retrieve TTL data
    bonsai_ttl, imec_TTL = retrieve_TTL_signals(session, TTL_bin_path)
    logger.info("The length of the bonsai TTL is: {} and the imec TTL is: {}".format(len(bonsai_ttl), len(imec_TTL)))
    assert len(imec_TTL) > len(bonsai_ttl), "Bonsai TTL is longer than imec TTL this can't be"

    #Check and correct for abberant signals
    imec_TTL, bonsai_ttl = check_for_abberant_signals(bonsai_ttl, imec_TTL, sampling_rate)

    #Get onset and offsets
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(bonsai_ttl, 2.5)
    ephys_sync_onsets, ephys_sync_offsets   = get_onset_offset(imec_TTL, 45)

    # Check to see that imec pulse onset is first 
    # ??What does this do?
    # if ephys_sync_onsets[0] >= bonsai_sync_onsets[0]:
    #     logger.error("The first pulse onset is the bonsai signal and not the imec signal. Error.")
    #     return

    # Check pulse lengths
    check_for_abberant_pulses(bonsai_sync_onsets, ephys_sync_onsets, sampling_rate)

    # remove pulses that are too brief
    bonsai_pulse_len_errors = np.where(np.diff(bonsai_sync_onsets) < sampling_rate / 3)[0]
    if bonsai_pulse_len_errors:
        logger.error(f"There is a pulse length error in the bonsai signal: {bonsai_pulse_len_errors}, removing pulse")
        bonsai_sync_offsets = np.delete(bonsai_sync_offsets, bonsai_pulse_len_errors)
        bonsai_sync_onsets  = np.delete(bonsai_sync_onsets, bonsai_pulse_len_errors)
    
    imec_pulse_len_errors = np.where(np.diff(ephys_sync_onsets) < sampling_rate / 3)[0]
    if imec_pulse_len_errors:
        logger.error(f"There is a pulse length error in the imec signal: {imec_pulse_len_errors}, removing pulse")
        ephys_sync_onsets = np.delete(ephys_sync_onsets, imec_pulse_len_errors)
        ephys_sync_offsets = np.delete(ephys_sync_offsets, imec_pulse_len_errors)
        
    if not imec_pulse_len_errors and bonsai_pulse_len_errors:
        logger.error("There was an imbalance in pulse lengths this can't be good")
        
    # Test that onsets len match
    assert len(bonsai_sync_onsets) == len(ephys_sync_onsets), f"The number of efizz pulses {len(ephys_sync_onsets)} onsets should match the number of bonsai pulses {len(bonsai_sync_onsets)} onsets."

    # define the TTL object
    ttl_object = TTL_Sync(bonsai_TTL = bonsai_ttl, 
                          imec_TTL = imec_TTL,
                          sampling_rate = sampling_rate,
                          bonsai_sync_onsets = bonsai_sync_onsets,
                          bonsai_sync_offsets = bonsai_sync_offsets,
                          ephys_sync_onsets = ephys_sync_onsets,
                          ephys_sync_offset = ephys_sync_offsets)
    
    return (ttl_object)

#--------------------------------------- Load and retrieve data functions -----------------------------------------------------------------------

# Retrieve TTL signals
def retrieve_TTL_signals(session: NEW_Session, TTL_bin_path: str):
    """Retrieves TTL signals for both the bonsai machine and the imec machine

    Args:
        session (Session): _description_

    Returns:
    - TTL signals from bonsai machine and imec board
    """
    #Retrieve TTL data
    
    AI_file = list(session.file_path.glob("*analog.bin"))[0] # need lst and idx as its a generator

    if '.bin' in str(AI_file): 
        AI_data = np.fromfile(AI_file)
        
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
        
    bonsai_ttl = AI_data[np.arange(3, len(AI_data), 4)] #From the 4 index until the end select every 4th sample
    imec_TTL = get_TTL_from_imec(TTL_bin_path)
    return bonsai_ttl, imec_TTL

#Load imec bin file
def get_TTL_from_imec(filename: str):
    """Load the imec bin file and convert to np memory map

    Args:
        filename (str): File name of .bin imec file produced by spikeGLX
    """
    data = load_or_open(filename, "int16", order="F", dtype="int16")
    return(data)

# ---------------------------------------------- utils ------------------------------------------------------------------

#Get the onsets and offsets for bonsai / imec. Your choice!
def get_onset_offset(signal, threshold, clean = True):
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
        logger.warning("Adding to the signal")
        starts = np.concatenate([[0], starts])

    #If the signal ends at the top of the pulse add the length of the signal to the end
    remove_last_onset = False
            
    if above[-1] > 0:
        logger.warning("Adding to the signal")
        ends = np.concatenate([ends, [len(signal)]])
        
        # In the event that the signal ends on a high, remove that last pulse
        # This is a hacky solution to the problem of the last pulse ending on a high as in for seq1
        # buffering should be fixed to prevent this
        # TODO: remove this if future recordings are affected
        remove_last_onset = True

    #If clean is true
    if clean:
        # offsets before the first onsets are removed
        ends = np.array([e for e in ends if e > starts[0]])

        # onsets before the last offsets are removed
        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    #If there aren't any starts or ends create empty arrays
    if not np.any(starts):
        starts = np.array([0])
        logger.error("No onsets")
    if not np.any(ends):
        ends = np.array([len(signal)])
        logger.error("No offsets")
    
    # If the last pulse is to be removed because of hack above, remove it
    if remove_last_onset:
        starts = starts[:-1]
        
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

    # check for signal differences, they should not differ by 10 seconds - Removing this check for now as it could be just a mannual delay between bonsai and spikeglx
    # if the user delays between stopping both systems
    if abs(len(bonsai_ttl) - len(imec_TTL)) > 10 * sampling_rate:
        logger.warning("The sync signals have very different lengths before resample, this cant be!")
        # raise ValueError("The sync signals have very different lengths before resample, this cant be!")

    # check for aberrant signals in ephys
    imec_errors = np.where(imec_TTL > 75)[0]
    if len(imec_errors):
        logger.warning(f"Found {len(imec_errors)} samples with too high values in probe signal")
    if len(imec_errors) > 1000:
        return False, 0, 0, "too_many_errors_in_ephys_sync_signal"

    # check of abberaant signals in bonsai TTL
    errors_bonsai = np.where(bonsai_ttl > 10)[0]
    assert len(errors_bonsai) < threshold, "There are too many abberant signals in the bonsai TTL"

    # If errors remove signals and update signals
    if errors_bonsai or imec_errors:
        imec_TTL = np.delete(imec_TTL, imec_errors)
        bonsai_ttl = np.delete(bonsai_ttl, errors_bonsai)
        logger.warning("Removing {} abberant signals from imec and {} from bonsai".format(len(imec_errors), len(errors_bonsai)))
    
    # Log success
    logger.success("Bonsai and Imec TTL are of similar lengths and have passed the abberant signal verification ")

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
    
    # log
    logger.info("Checking if pulse lenghts are as expected")

    #Bonsai pulse length check if too long or short
    bonsai_pulse_len_under_errors = np.where(np.diff(bonsai_sync_onsets) < (sampling_rate / 2))[0] # Is onset delta less than 15khz
    bonsai_pulse_len_over_errors = np.where(np.diff(bonsai_sync_onsets) > (sampling_rate * 1.5))[0] # Is onset delta greater than 15khz

    if bonsai_pulse_len_under_errors:
        onsets_delta = np.diff(bonsai_sync_onsets)
        counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
        logger.warning(f"Bonsai pulse less than 1hz duration: {counts}")

    if bonsai_pulse_len_over_errors:
        logger.warning("Bonsai pulse greater than 1hz duration")

    #Imec pulse length check if too long or short
    imec_pulse_len_under_errors = np.where(np.diff(ephys_sync_onsets) < (sampling_rate / 2))[0] # Is onset delta less than 15khz
    imec_pulse_len_over_errors  = np.where(np.diff(ephys_sync_onsets) > (sampling_rate * 1.5))[0] # Is onset delta greater than 15khz

    if imec_pulse_len_under_errors:
        logger.error("Imec pulse less than expected 1hz duration")

    if imec_pulse_len_over_errors:
        logger.warning("Bonsai pulse greater than 1hz duration")