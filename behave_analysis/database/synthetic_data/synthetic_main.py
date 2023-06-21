# Created dataframe has unused columns but required due to indexing in visualize efizz.
# TODO: remove unused columns from dataframe when indexing is fixed in efizz to be polars 

from databank import experiments_objects
from behave_analysis.process.process import Process
from dataclasses import dataclass
import polars as pl
import os
import dill as pickle
import numpy as np 
from loguru import logger
import matplotlib.pyplot as plt

# Collect session data currently in the databank
for session_ID in experiments_objects:
    session = Process(session_ID).load_session()
    break

def synthetic_dataframe(tuning):
    
    np.random.seed(42)  # For reproducibility, you can remove this line for true randomnes
    cell_num = 37 # how many cells to generate per type
    session, tracking = load_tracking_data()
    spikes_per_clu = efizz_stats(session)

    all_spikes = []
    all_clu_ID = []
    all_clu_label = []
    clu_offset = 0

    for cell_type in tuning:
        
        if cell_type == "hdir": 
            angles = tracking["hdir"]
        elif cell_type == "hsa": 
            angles = tracking["hdir_shelt"]
        elif cell_type == "h_bar_north_a": 
            angles = tracking['hdir_barrier'][:,0]
        elif cell_type == "h_bar_south_a": 
            angles = tracking['hdir_barrier'][:,1]
        
        # Generate vectorial cells 
        spikes, clusters = return_spike_times_locked_to_behavioural_direction(angles, 
                                                                            number_of_spikes = np.random.choice(spikes_per_clu, size = cell_num, replace = False), 
                                                                            direction = np.linspace(-np.pi,np.pi,cell_num), 
                                                                            dir_std = np.random.choice(np.linspace(.1,np.pi/4,50), size = cell_num),
                                                                            fps = 40,
                                                                            mean = 0, 
                                                                            std_dev = 1,
                                                                            add_noise = False,
                                                                            noise_scale = 25)
        
        all_spikes.extend(spikes)
        all_clu_ID.extend(clusters + clu_offset)
        clu_offset = clu_offset + np.amax(clusters)
        all_clu_label.extend([cell_type]*len(spikes))
    
    all_spikes = np.array(all_spikes)
    all_clu_ID = np.array(all_clu_ID)

    # Create a Polars DataFrame
    synth_df = pl.DataFrame({"spike_times": [0] * len(all_spikes), # Not used
                            "spike_clusters": all_clu_ID,
                            "cluster_group": all_clu_label,
                            "aligned_spike_times": all_spikes,
                            "aligned_spike_times_in_samples": [0] * len(all_spikes), # Not used
                            "spike_aligned_to_frame": np.around((all_spikes * 40))})
    
    synth_df = synth_df.with_columns(synth_df["spike_aligned_to_frame"].cast(pl.Float64))

    return synth_df

def return_spike_times_locked_to_behavioural_direction(behavioural_direction, 
                                                       number_of_spikes,
                                                       direction, 
                                                       dir_std = .1,
                                                       fps = 40, 
                                                       mean = 0, 
                                                       std_dev = 1,
                                                       add_noise = False,
                                                       noise_scale=0.1) -> np.ndarray:
    
    """
    Selecting the behavioural direction of interest, this function will gather the samples
    of when that direction is occuring, convert into time and then generate spikes around that time. 
    """
    all_spike_times = []
    all_cluster_ids = []
    
    for idx, z in enumerate(zip(direction,number_of_spikes,dir_std), start=1):
        direction = z[0]
        spike_num = z[1]
        dir_std = z[2]
        
        indices = np.where(((direction - dir_std) <= behavioural_direction) & (behavioural_direction <= (direction + dir_std)))[0]
        times = indices / fps
        spike_times = generate_spikes(spike_num, times, mean, std_dev)
    
        if add_noise:
                noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=len(spike_times))
                spike_times += noise
                
        all_spike_times.extend(spike_times)
        all_cluster_ids.extend([idx] * len(spike_times))
    
    all_spike_times = np.array(all_spike_times)
    all_cluster_ids = np.array(all_cluster_ids)
                
    return all_spike_times, all_cluster_ids

## UTIL FUNCTIONS ----------------------------------------------------
def load_tracking_data():
    """Just for first session, load the tracking data."""
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        break
    file = os.path.join(session.processed_path, "fully_processed_tracking_data.pickle")
    with open(file, "rb") as dill_file:
        tracking = pickle.load(dill_file)
    return session, tracking

def efizz_stats(session):
    spikedataframe = session.efizzDataProcessed.alignedDataFrame.filter((session.efizzDataProcessed.alignedDataFrame['cluster_group'] == "good")
                                                                        | (session.efizzDataProcessed.alignedDataFrame['cluster_group'] == "mua"))
    spikecount = spikedataframe.groupby("spike_clusters").count()
    number_of_spikes = spikecount["count"].to_numpy()
    return number_of_spikes

def generate_spikes(spike_count, 
                    times, 
                    mean, 
                    std_dev) -> np.ndarray:
    """
    Produces a gaussian of spikes around times of behavioural relevant events.
    """
    valid_indices = np.arange(len(times))
    
    if len(valid_indices) == 0:
        raise ValueError("No valid indices to choose from")
    
    chosen_indices = np.random.choice(valid_indices, size=spike_count)
    gaussian_offsets = np.random.normal(mean, std_dev, size=spike_count)
    spike_times = times[chosen_indices] + gaussian_offsets
    spike_times.sort()
    return spike_times