#OS Libraries
import numpy as np
from loguru import logger

#Custom libs
from behave_analysis.process.ttl_sync import derivative

def remove_idx_as_per_bonsai_ttl_resample(name_of_signal, signal_to_downsample, indexs_to_remove, temporal_diff):
    copy_of_original_signal_for_test = np.copy(signal_to_downsample) # Copy signal to conduct length test
    down_sampled_signal = np.delete(signal_to_downsample, indexs_to_remove) # Delete that index - all at once
    logger.info("{}: Signal downsampled to match bonsai TTL resample".format(name_of_signal))

    #Tests
    assert (len(down_sampled_signal) == len(copy_of_original_signal_for_test) - sum(temporal_diff)), "The new signal does not match the old - number of changes required" 
    assert all(derivative(indexs_to_remove) > 1000), "A re_sample is less than 1000 samples apart. Not uniform "
    return(down_sampled_signal)