"""
Refactor: Where the efizz data comes from

"""

# Custom Libaries
from __future__ import annotations
from itertools import count
from behave_analysis.process.session import Session
from behave_analysis.utils.mat_to_python import convert_matlab_struct
from databank import efizz

# OS Libaries
from dataclasses import dataclass
from loguru import logger
import numpy as np
import pandas as pd

#Data
ephys_file_path = efizz["1677_NoShelterShelter_22MAY31"]["res"]

@dataclass(frozen=True)
class Ephys:
    spike_times: object
    cluster_ids: object
    spike_mask: object
    num_spikes: int
    spike_dic: object
    annotations: object

def get_Ephys(session: Session):

    # Load spike times and cluster ids, and define total spike count
    spike_times, cluster_ids, annotations = load_ephys_data(ephys_file_path)
    num_spikes = len(spike_times)

    # Align spikes times to pulse onset
    offset = session.ttl.imec_delay
    spike_times, indexes_removed = offset_spike_times(offset, spike_times)

    # Update spike clude id index
    cluster_ids = cluster_ids[indexes_removed:]
    assert len(cluster_ids) == len(spike_times), "cluster ids should match spike len"
    
    # spike data not resampled as can't make fast function so commenting out, meaning data slightly off
    
    if session.ttl.idxs_2_remov_from_imec_sig:
        # Resample spike data to remove indexes that might of by chance removed from resample alignment
        cluster_ids, spike_times = resample_spike_data(cluster_ids,
                                                    spike_times,
                                                    session.ttl.idxs_2_remov_from_imec_sig)

    # create spike dic
    spike_dic = create_spike_dic(cluster_ids, spike_times)

    # Create spike mask
    spike_mask = create_spike_mask(spike_times, session.ttl.imec_TTL)
    
    ephys = Ephys(spike_times,
                  cluster_ids,
                  spike_mask,
                  num_spikes,
                  spike_dic,
                  annotations)
                
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
    annotations = data.dictionary.clusterNotes
    logger.info("Number of spike times: {}".format(len(spike_times)))
    assert len(spike_times) == len(cluster_ids), "The length of spike times and cluster ids should match"
    return spike_times, cluster_ids, annotations

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
        
        # spike_mask = np.put(spike_mask, spike_times, 1)
        
        # Assertions and loggs
        assert spike_mask[spike_times[0]] == 1, "The first spike time should equal 1"
        assert spike_mask[spike_times[-1]] == 1, "The last spike time should equal 1"

        return spike_mask
    
def resample_spike_data(cluster_ids,
                        spike_times,
                        indexs_removed_from_imec):
    """A function that removes indexes from the spike data that were removed
    from the Imec TTL signal to ensure Imec and bonsai are aligned.

    Args:
        cluster_ids (_type_): _description_
        spike_times (_type_): _description_

    Returns:
        _type_: _description_
        
    Refactor:
    - Very slow. Factorise cluster id if possible but spike time calc also slow
    - not used as so slow meaning spike data not resampled and slightly misaligned
    """
    
    logger.warning("Resampling spike data as Imec signal required it")

    spike_times_data_frame = pd.DataFrame(spike_times, columns = ["spike_times"])
    output = spike_times_data_frame.spike_times.isin(indexs_removed_from_imec)
    idx = np.asarray(output[output].index) # Retrieve index and convert to array
    
    # Use those idxs to remove from both cluster id and spike times
    spike_times = np.delete(spike_times, idx)
    cluster_ids = np.delete(cluster_ids, idx)
    
    # Assertions to check length
    assert len(cluster_ids) == len(spike_times), "The Lengths of these two should match"
    
    return cluster_ids, spike_times

def offset_spike_times(offset,
                       spike_times) -> object:
    """A function that removes the spikes occuring before the
    first imec pulse onset and reduces all spikes after the first onset
    by the difference between the start of the imec signal and the first imec pulse
    onset to ensure the spike data is aligned with the analogue data.

    Args:
        offset (int): delta between imec start and first pulse onset
        spike_times (obect): array of spike indexes

    Returns:
        spike_times: aligned spike times
    """
    initial_len = len(spike_times)
    spike_times = spike_times - offset
    spike_times = np.where(spike_times >= 0, spike_times, None)
    spike_times = spike_times[spike_times != None]
    resulting_len = len(spike_times)
    indexes_removed = initial_len - resulting_len

    return spike_times, indexes_removed

def create_spike_dic(cluster_ids, spike_times):
    """A function that creates a dictionary where the keys are spike clusters and 
       the values are the indexes aligning to spike times

    Args:
        cluster_ids (object): Array of spike clusters
        spike_times (object): Array of spike indexes

    Returns:
        Dictionary: keys are spike clusters and values are spike indexes assigned to that cluster
    """
    cluster_id_max = max(cluster_ids)
    spike_dic = {}
    for cluster in range(cluster_id_max + 1):
        bool = cluster_ids == cluster # What indexes match this cluster, True or False
        indexes = np.where(bool)[0] # extract the indexes
        spike_dic[cluster] = list(np.take(spike_times, indexes)) # Assign those indexes and extract spike times to the dic
    
    # Assertions
    flat_list = [x for xs in spike_dic.values() for x in xs]
    assert len(flat_list) == len(spike_times), "The length of values of the spike dic should match the total spike times"
    return spike_dic