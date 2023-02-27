# OS Libraries
import numpy as np
from loguru import logger

# Custom libs
from behave_analysis.process.electrophysiology.ttl_sync import derivative


def remove_idx_as_per_bonsai_ttl_resample(name_of_signal,
                                          signal_to_downsample,
                                          indexs_to_remove,
                                          session):
    """A function that takes in the indexes removed from the bonsai signal and removes the same from other
    AI data streams such as the camera trigger, the audio and the photoresistor.

    refactor:
    You have to generate off sets and onsets again to get correct index after cleaning so a bit clunky,
    need to refactor as this code is different location to one used in tt_sync

    Args:
        name_of_signal (str): what is the name of the signal downsampled
        signal_to_downsample (int): array of analogue signal
        indexs_to_remove (int): array of indexes to remove
        session data to retrieve bonsai TTL

    return:
        Downsampled signal with the start and ends cutt off to match bonsai
    """
    
    assert len(signal_to_downsample) == session.ttl.inital_bonsai_len, "Length of analogue signals should match bonsai signal"

    down_sampled_signal = np.delete(signal_to_downsample, indexs_to_remove)

    # clean start
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(session.ttl.bonsai_obj['post resample'], 2.5)
    cleaned_signal = remove_start_of_signal(down_sampled_signal, bonsai_sync_onsets)

    # clean end
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(session.ttl.bonsai_obj['post start of signal cut'], 2.5)
    cleaned_signal = remove_end_of_signal(cleaned_signal, bonsai_sync_offsets)

    # Tests
    try:
        assert len(cleaned_signal) == len(session.ttl.bonsai_TTL), "The downsampled analogue data does not match the length of the bonsai signal"
        logger.info(f"{name_of_signal}: Signal downsampled to match bonsai TTL resample. Was {len(signal_to_downsample)} and now {len(cleaned_signal)} ")
    except:
        print(len(cleaned_signal))
        print(len(len(session.ttl.bonsai_TTL)))
        print("Down sample failed")
    return(cleaned_signal)


def remove_start_of_signal(signal_to_clean, bonsai_sync_onsets):
    """A function that removes the signal before the first onset

    Args:
        signal_to_clean (object): of type audio, video etc.
        bonsai_sync_onsets (obect): when does the bonsai TTL pulse onset

    Returns:
        _type_: _description_
    """
    # Get first index
    bonsai_first_pulse_idx = bonsai_sync_onsets[0]

    # remove start of signal
    signal_to_clean = signal_to_clean[bonsai_first_pulse_idx:]

    return signal_to_clean

# Remove the flat line end at the end of the imec signal


def remove_end_of_signal(signal_to_clean, bonsai_sync_offsets):
    """A function that removes the signal after the last offset

    Args:
        signal_to_clean (object): of type audio, video etc.
        bonsai_sync_onsets (obect): when does the bonsai TTL pulse last offset

    Returns:
        _type_: _description_
    """
    # Get last offset
    bonsai_last_pulse_idx = bonsai_sync_offsets[-1]

    # remove start of signal
    cleaned_signal = signal_to_clean[:bonsai_last_pulse_idx]

    return cleaned_signal

# Get the onsets and offsets for bonsai / imec. Your choice!


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
    above = np.zeros_like(signal)  # Creates an array of zeros of length signal
    # If the signal is above voltage threshold, set to 1
    above[signal >= threshold] = 1
    der = derivative(above)  # Create an array of differences
    starts = np.where(der > 0)[0]  # Where does the signal switch from 0 to 1
    ends = np.where(der < 0)[0]  # Where does the signal switch from 1 to 0

    # If the signal starts with a pulse add a zero to the start
    if above[0] > 0:
        starts = np.concatenate([[0], starts])

    # If the signal ends at the top of the pulse add the length of the signal to the end
    if above[-1] > 0:
        ends = np.concatenate([ends, [len(signal)]])

    # If clean is true
    if clean:
        # ends before the first start are removed
        ends = np.array([e for e in ends if e > starts[0]])

        # starts before the last end are removed
        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    # If there aren't any starts or ends create empty arrays
    if not np.any(starts):
        starts = np.array([0])
        logger.error("No onsets")
    if not np.any(ends):
        ends = np.array([len(signal)])
        logger.error("No offsets")
    return starts, ends

# Take the derivate so you can spot changes in state within a signal. Ie, pulse onset/offsets

def derivative(X, axis=0, order=1):
    """"Takes the derivative of an array X along a given axis

        Arguments:
            X: np.array with data - 1 dimensional
            axis: int. Axis along which the derivative is to be computed
            order: int. Derivative order
    """
    # Prepend 0 so the index is realigned to prevent off by 1 error
    return np.diff(X, n=order, axis=axis)
