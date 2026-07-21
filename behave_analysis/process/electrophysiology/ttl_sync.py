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

# Custom libaries
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.load_bin_or_np import load_or_open
from behave_analysis.utils.AI_dataClass_objects import TTL_Sync

# OS libaries
import os
import numpy as np
from glob import glob
import dill as pickle
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

# Globals
sampling_rate = 30000

def get_TTL(session: NEW_Session, TTL_bin_path: str):
    """Returns the TTL_sync dataclass.

    Args:
        session (Session): custom object containing experimental path file
        TTL_bin_path (str): path to the TTL bin file exported from spikeglx, this is the imec sync signal

    Returns:
        TTL_Sync: data class
    """
    
    bonsai_ttl, imec_TTL = retrieve_TTL_signals(session, TTL_bin_path)
    
    logger.info("The length of the bonsai TTL is: {} and the imec TTL is: {}".format(len(bonsai_ttl), len(imec_TTL)))
    # assert len(imec_TTL) > len(bonsai_ttl), "Bonsai TTL is longer than imec TTL this can't be - the session likely crashed or disconnected"
    imec_TTL, bonsai_ttl = check_for_abberant_signals(bonsai_ttl, imec_TTL, sampling_rate)

    # Extract the onset and offsets for the TTL signals and check they match -----------------------------------
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(bonsai_ttl, 2.5)
    ephys_sync_onsets, ephys_sync_offsets = get_onset_offset(imec_TTL, 0.5)

    # Check pulse lengths
    (
        bonsai_sync_onsets,
        bonsai_sync_offsets,
        ephys_sync_onsets,
        ephys_sync_offsets,
    ) = check_for_abberant_pulses(
        bonsai_sync_onsets,
        bonsai_sync_offsets,
        ephys_sync_onsets,
        ephys_sync_offsets,
        sampling_rate,
    )
    
    if (session.mouse == "JAL006") and (session.date == "2024_04_01"):
        # Hacky logic for JAL6 April 1st session
        # Step 1: Remove the assertion to ensure imec is longer
        # Step 2: Select the same number of onsets for both
        diff = len(bonsai_sync_onsets) - len(ephys_sync_onsets)
        bonsai_sync_onsets = bonsai_sync_onsets[diff:] # chop the beginning?
    if (session.mouse == "JAL007") and (session.date == "2024_04_04"):
        # for JAL7 4april, disconnected, chop the end
        bonsai_sync_onsets = bonsai_sync_onsets[:len(ephys_sync_onsets)]
        bonsai_sync_offsets = bonsai_sync_offsets[:len(ephys_sync_onsets)]
    # visualize alignment
    # i = 0
    # plt.plot(imec_TTL[ephys_sync_onsets[i]-10000:ephys_sync_onsets[i]+150000])
    # plt.plot(bonsai_ttl[bonsai_sync_onsets[i]-10000:bonsai_sync_onsets[i]+150000])
    # plt.show()
    # i = len(ephys_sync_onsets) - 1
    # plt.plot(imec_TTL[ephys_sync_onsets[i]-150000:ephys_sync_onsets[i]+150000])
    # plt.plot(bonsai_ttl[bonsai_sync_onsets[i]-150000:bonsai_sync_onsets[i]+150000])
    # plt.show()
    
    assert len(bonsai_sync_onsets) == len(ephys_sync_onsets), f"The number of efizz pulses {len(ephys_sync_onsets)} onsets should match the number of bonsai pulses {len(bonsai_sync_onsets)} onsets."
    logger.success("The number of efizz pulses onsets match the number of bonsai pulses onsets")

    # define the TTL object
    ttl_object = TTL_Sync(
        bonsai_TTL=bonsai_ttl,
        imec_TTL=imec_TTL,
        sampling_rate=sampling_rate,
        bonsai_sync_onsets=bonsai_sync_onsets,
        bonsai_sync_offsets=bonsai_sync_offsets,
        ephys_sync_onsets=ephys_sync_onsets,
        ephys_sync_offset=ephys_sync_offsets,
    )

    # save photoresistor
    meta_file = os.path.join(session.base_path,session.processed_path,'TTL_file')
    with open(meta_file, "wb") as dill_file:
        pickle.dump(ttl_object, dill_file)

    return ttl_object


# --------------------------------------- Load and retrieve data functions -----------------------------------------------------------------------


# Retrieve TTL signals
def retrieve_TTL_signals(session: NEW_Session, TTL_bin_path: str):
    """Retrieves TTL signals for both the bonsai machine and the imec machine

    Args:
        session (Session): _description_

    Returns:
    - TTL signals from bonsai machine and imec board
    """
    # Retrieve sync pulse from bonsai NIDAQ file -----------------------------------------------
    full_file_path = Path(os.path.join(session.base_path, session.file_path))
    AI_file = list(full_file_path.glob("*analog.bin"))[0]  # need lst and idx as its a generator

    if ".bin" in str(AI_file):
        AI_data = np.fromfile(AI_file)

    else:
        with open(AI_file, "rb") as dill_file:
            AI_data = pickle.load(dill_file)

    bonsai_ttl = AI_data[np.arange(3, len(AI_data), 4)]  # From the 4 index until the end select every 4th sample

    # Retrieve sync pulse from imec spikeglx file -----------------------------------------------
    if 'exported' in TTL_bin_path:
        imec_TTL = unpackbits(np.fromfile(Path(TTL_bin_path), dtype=np.int16), bit_filter=6)
    else:
        logger.warning("The TTL sync channel is not in an exported .bin file! Using the old method of extracting from ap.bin!")
        # for NPX1 or other weird cases - the old way to load the sync channel  
        imec_TTL = get_TTL_from_imec(TTL_bin_path)

    return bonsai_ttl, imec_TTL


def unpackbits(exported_sync_channel_imec, num_bits=16, bit_filter=6):
    """This is a yulin wizard function that extracts the imec digital signal from the
    exported imec file. You must use spikeglx viewer to export only the 384 sync channel.
    It unpacks the bin file in bits and returns a digital signal of 0s and 1s.
    """
    xshape = list(exported_sync_channel_imec.shape)
    x = exported_sync_channel_imec.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    temp = (x & to_and).reshape(xshape + [num_bits])
    if bit_filter is not None:
        temp = temp[:, 6]
    temp[temp > 0] = 1
    digital_signal = temp
    return digital_signal


# ---------------------------------------------- utils ------------------------------------------------------------------


# Testing new onset to see if it works with less functionality
def get_onset_offset(signal, threshold, clean=True):
    """
    Get onset/offset times when a signal (either bonsai or imec TLL depending on argument)
    goes below>above and above>below a given threshold. If no starts or ends kill programme.

    Arguments:
        signal: 1d numpy array - bonsai or imec TLL
        thhreshold: float, threshold
        clean: bool. If true ends before the first start and starts after the last end are removed
        type: imec or bonsai

    Returns:
        Starts: Indexes of pulse onsets
        Ends: Indexes of pulse offsets
    """

    above = np.zeros_like(signal)
    above[signal >= threshold] = 1  # If the signal is above voltage threshold, set to 1
    if np.sum(np.isnan(signal)) > 0:
        above[np.isnan(signal)] = np.nan
    der = np.diff(above, n=1, axis=0)  # Create an array of differences
    starts = np.where(der > 0.5)[0]  # Where does the signal switch from 0 to 1
    ends = np.where(der < -0.5)[0]  # Where does the signal switch from 1 to 0

    if clean:

        # offsets before the first onsets are removed
        ends = np.array([e for e in ends if e > starts[0]])

        # onsets before the last offsets are removed
        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    if not np.any(starts):
        assert False, "No onsets"
    if not np.any(ends):
        assert False, "No offsets"

    return starts, ends


# -------------------------------------- Abberant signal checks --------------------------------------------------------------------------


def check_for_abberant_signals(bonsai_ttl, imec_TTL, sampling_rate):
    """A function that checks:
    1) If the signal lengths are too different between bonsai and immec, they shouldn't be longer than
    10 seconds. Enabling bonsai and disabling bonsai and spike glx shouldn't take >10s but if mannually this occured
    then this error would fire, or if another issue occured
    2) If the signals deviate away from an expected baseline
    3) If the signals are too high for example, then just remove those signals just for the alignment

    Note if there are too many abberant signals this function will fail the assert.

    Args:
        bonsai_ttl (object): TTL signal
        imec_TTL (object): TTL signal
        sampling_rate (int): FS of sampling system

    Returns:
        Object: Imec TTL Signal cleaned from abberant spikes
        Object: bonsai TTL signal cleaned from abberant spikes
    """

    # Threshold for acceptable number of abberant signals
    threshold = len(bonsai_ttl) * 0.1  # this seems arbitrary

    # check for signal differences, they should not differ by 30 seconds. Unless there has been a mannual delay between stopping both systems. 
    if abs(len(bonsai_ttl) - len(imec_TTL)) > 30 * sampling_rate:
        logger.warning("The sync signals are more than 30 seconds different. Either there has been a mannual delay between stopping both systems or there is an error in the data.")

    # check for aberrant signals in ephys
    imec_errors = np.where(imec_TTL > 80)[0]
    if len(imec_errors) > 0:
        logger.warning(f"Found {len(imec_errors)} samples with too high values in probe signal")

    # check of abberaant signals in bonsai TTL
    errors_bonsai = np.where(bonsai_ttl > 5.1)[0]
    if len(errors_bonsai) > 0:
        logger.warning(f"Found {len(errors_bonsai)} samples with too high values in bonsai signal")
        assert len(errors_bonsai) < threshold, "There are too many abberant signals in the bonsai TTL"

    # If errors remove signals and update signals
    if len(imec_errors) > 0:
        imec_TTL = imec_TTL.astype(float)
        imec_TTL[imec_errors] = np.nan
        imec_TTL[imec_errors] = [np.nanmean([imec_TTL[i - 5 : i - 1], imec_TTL[i + 1 : i + 5]]) for i in imec_errors]
        logger.warning("Removing {} abberant signals from imec".format(len(imec_errors)))

    if len(errors_bonsai) > 0:
        bonsai_ttl = bonsai_ttl.astype(float)
        bonsai_ttl[errors_bonsai] = np.nan
        bonsai_ttl[errors_bonsai] = [np.nanmean([bonsai_ttl[i - 5 : i - 1], bonsai_ttl[i + 1 : i + 5]]) for i in errors_bonsai]
        logger.warning("Removing {} abberant signals from bonsai".format(len(errors_bonsai)))

    # Log success
    logger.info("Abberant signal verification test finished")

    return imec_TTL, bonsai_ttl


# Highlight weird length pulses
def check_for_abberant_pulses(
    bonsai_sync_onsets,
    bonsai_sync_offsets,
    ephys_sync_onsets,
    ephys_sync_offsets,
    sampling_rate,
    delete=True,
):
    """A function that checks if the delta between onsets and offsets is not greater or less than
    what should roughly be expected. Are pulse lengths to be expected?

    Args:
        bonsai_sync_onsets (_type_): _description_
        bonsai_sync_offsets (_type_): _description_
        ephys_sync_onsets (_type_): _description_
        ephys_sync_offsets (_type_): _description_
        sampling_rate (_type_): _description_
        delete (bool): If True, remove the pulse onsets that are too brief, this is likely a result of a bad sync signal. This will hopefully fix alignment issues.
            if False, just log the error. 
            
            If syncing not working, FIRST SET TO FALSE, then check the onsets and offsets to see if they are correct.
    """

    # log
    logger.info("Checking if pulse lenghts are as expected")

    # Bonsai pulse length check if too long or short
    bonsai_pulse_len_under_errors = np.where(np.diff(bonsai_sync_onsets) < (sampling_rate / 2))[0]  # Is onset delta less than 15khz
    bonsai_pulse_len_over_errors = np.where(np.diff(bonsai_sync_onsets) > (sampling_rate * 1.5))[0]  # Is onset delta greater than 15khz

    if bonsai_pulse_len_under_errors.any():
        onsets_delta = np.diff(bonsai_sync_onsets)
        counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
        logger.warning(f"There are {len(bonsai_pulse_len_under_errors)} bonsai pulses that are less than 1hz duration")
        logger.warning(f"Bonsai pulse less than 1hz duration: {counts}")
        
        if delete:
            logger.warning("Removing bonsai pulses onsets that are too brief, this is likely a result of a bad sync signal. This will hopefully fix alignment issues.")
            bonsai_sync_onsets = np.delete(bonsai_sync_onsets, bonsai_pulse_len_under_errors)
            bonsai_sync_offsets = np.delete(bonsai_sync_offsets, bonsai_pulse_len_under_errors)

    if bonsai_pulse_len_over_errors.any():
        logger.warning("Bonsai pulse greater than 1hz duration")

    # Imec pulse length check if too long or short
    imec_pulse_len_under_errors = np.where(np.diff(ephys_sync_onsets) < (sampling_rate / 2))[0]  # Is onset delta less than 15khz
    imec_pulse_len_over_errors = np.where(np.diff(ephys_sync_onsets) > (sampling_rate * 1.5))[0]  # Is onset delta greater than 15khz

    if imec_pulse_len_over_errors.any():
        logger.warning("Bonsai pulse greater than 1hz duration")
        
    if imec_pulse_len_under_errors.any():
        onsets_delta = np.diff(ephys_sync_onsets)
        counts = {k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)}
        logger.warning(f"There are {len(imec_pulse_len_under_errors)} imec pulses that are less than 1hz duration")
        logger.warning(f"Imec pulses less than 1hz duration: {counts}")
        
        if delete:
            logger.warning("Removing imec pulses onsets that are too brief, this is likely a result of a bad sync signal. This will hopefully fix alignment issues.")
            ephys_sync_onsets = np.delete(ephys_sync_onsets, imec_pulse_len_under_errors)
            ephys_sync_offsets = np.delete(ephys_sync_offsets, imec_pulse_len_under_errors)
    
    logger.info("Pulse checks complete. Onsets and offsets were filtered together where needed.")

    return bonsai_sync_onsets, bonsai_sync_offsets, ephys_sync_onsets, ephys_sync_offsets

# ====================================================================================================================================================================

# Test old functions

#Load imec bin file

# NOTE - This is an old function that is not used in the current pipeline.
# I brought it back here to test syncing of a broken session and it worked so leaving it here in case it is needed in the future.
def get_TTL_from_imec(filename: str):
    """Load the imec bin file and convert to np memory map

    Args:
        filename (str): File name of .bin imec file produced by spikeGLX
    """
    data = load_or_open(filename, "int16", order="F", dtype="int16")
    return(data)


