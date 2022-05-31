#OS Libraries
import numpy as np
from loguru import logger

#Custom libs
from behave_analysis.process.ttl_sync import derivative

def remove_idx_as_per_bonsai_ttl_resample(name_of_signal, signal_to_downsample, indexs_to_remove, temporal_diff):
    """A function that takes in the indexes removed from the bonsai signal and removes the same from other
    AI data streams such as the camera trigger, the audio and the photoresistor.

    Args:
        name_of_signal (str): what is the name of the signal downsampled
        signal_to_downsample (int): array of analogue signal
        indexs_to_remove (int): array of indexes to remove
        temporal_diff (int): number of samples to remove. Different in dif between onsets 

    return:
        Downsampled signal
    """
    copy_of_original_signal_for_test = np.copy(signal_to_downsample) # Copy signal to conduct length test
    down_sampled_signal = np.delete(signal_to_downsample, indexs_to_remove) # Delete that index - all at once
    logger.info("{}: Signal downsampled to match bonsai TTL resample".format(name_of_signal))

    #Tests
    assert (len(down_sampled_signal) == len(copy_of_original_signal_for_test) - sum(temporal_diff)), "The new signal does not match the old - number of changes required" 
    return(down_sampled_signal)
