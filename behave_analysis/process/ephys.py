# Custom Libaries
from behave_analysis.process.session import Session
from behave_analysis.utils.mat_to_python import convert_matlab_struct

# OS Libaries
from dataclasses import dataclass
from loguru import logger
import numpy as np 

#Data
ephys_file_path = r"D:\Electrophysiology_data\1677_ObstacleThenRemove_22MAY23_g0\1677_ObstacleThenRemove_22MAY23_g0_imec0\1677_ObstacleThenRemove_22MAY23_g0_t0.imec0.ap_res.mat"

@dataclass(frozen=True)
class Ephys:
    spike_times: object
    cluster_ids: object
    spike_mask: object
    num_spikes: int

def get_Ephys(session: Session):

    #Load spike times and cluster ids, and define total spike count
    spike_times, cluster_ids = load_ephys_data(ephys_file_path)
    num_spikes = len(spike_times)

    # Create spike mask
    spike_mask = create_spike_mask(spike_times, session.ttl.imec_TTL)

    ephys = Ephys(spike_times,
                  cluster_ids,
                  spike_mask,
                  num_spikes)
                
    return ephys

def load_ephys_data(ephys_file_path):
    """A function that collects the efizz data and converts from matlab to python

    Need to refactor the file location of the efizz data

    Returns:
    - spike times: object
    - cluster ids: object 
    """
    data = convert_matlab_struct(ephys_file_path)
    spike_times = data.dictionary.spikeTimes
    cluster_ids = data.dictionary.spikeClusters
    logger.info("Number of spike times: {}".format(len(spike_times)))
    assert len(spike_times) == len(cluster_ids), "The length of spike times and cluster ids should match"
    return spike_times, cluster_ids

def create_spike_mask(spike_times, imec_TTL):
        """A function that first creates a spike mask where one or more spikes = 1 and no spikes = 0.

        Returns:
            object:  Binary array of whether a spike occured or not
        """

        # Create a spike mask of spike counts across the session
        spike_mask = np.zeros(spike_times[-1] + 1) # Create empty array of zeros of lenght last spike index
        logger.info("The length of the spike mask is: {}".format(len(spike_mask)))

        #For each spike time, add 1 to spike count. If more spikes happened within one index add another spike
        #For testing purposes
        for spike in spike_times: 
            spike_mask[spike] = 1
        
        # Assertions and loggs
        assert spike_mask[spike_times[0]] == 1, "The first spike time should equal 1"
        assert spike_mask[spike_times[-1]] == 1, "The last spike time should equal 1"

        return spike_mask
