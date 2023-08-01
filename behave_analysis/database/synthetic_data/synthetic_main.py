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
from scipy import interpolate
from typing import Tuple

# Globals
np.random.seed(42)  # For reproducibility, you can remove this line for true randomnes

# Collect session data currently in the databank
for session_ID in experiments_objects:
    session = Process(session_ID).load_session()
    break

def extract_tuning_request(cell_type_to_generate, tracking_data):
    """
    First check that the requested angle is permitted. Then return 
    the behavioural angles for stimuli of interest.
    """
    options = ['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']
    if cell_type_to_generate not in options:
        raise ValueError
    
    if cell_type_to_generate == "hdir": 
        angles = tracking_data["hdir"]
    elif cell_type_to_generate == "hsa": 
        angles = tracking_data["hsa"]
    elif cell_type_to_generate == "h_bar_north_a": 
        angles = tracking_data['h_bar_north_a']
    elif cell_type_to_generate == "h_bar_south_a": 
        angles = tracking_data['h_bar_south_a']
    
    return angles
    
def generate_synthetic_dataframe(tuning: list, 
                                 realistic = True,
                                 num_cells_per_type = 37, 
                                 number_of_spikes_to_gen_per_cluster = 250000,
                                 pass_video_df = None) -> pl.DataFrame:
    """
    Inputs: 
    
    + tuning (type: list) - A list of strings. Each entry contains an angle and can range from a single angle ['hdir] to 4. E.g: ['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']
    + realistic (flag) - use real firing counts to generate data from efizz stats, or make unrealistic ones for sure bet testing of models
    + num_cells_per_type - How many different cluusters do you want per tuning category, the more cells the more angles that will be covered / generated for
    
    Returns: synth_df (type: pl.DataFrame)
    Description: Synthetic polars dataframe that attempts to match the real data
    """
    
    tracking = pass_video_df
    session, _ = load_tracking_data()
        
    if realistic: # NOTE real behavioural data is used regardless, this just controls efizz stats
        spikes_per_clu = efizz_stats(session)
        number_of_spikes_to_gen_per_cluster = np.random.choice(spikes_per_clu, size = num_cells_per_type, replace = False)
        
    if not realistic:
        number_of_spikes_to_gen_per_cluster = [number_of_spikes_to_gen_per_cluster] * num_cells_per_type
    
    # Init
    all_spikes = []
    all_clu_ID = []
    all_clu_label = []
    clu_offset = 0

    for cell_type in tuning:
        angles = extract_tuning_request(cell_type_to_generate = cell_type,   tracking_data = tracking)
        spikes, clusters = return_spike_times_locked_to_behavioural_direction(behavioural_direction = angles.to_numpy(), 
                                                                              number_of_spikes = number_of_spikes_to_gen_per_cluster,
                                                                              tuned_direction = np.linspace(-np.pi, np.pi, num_cells_per_type), 
                                                                              dir_std = np.random.choice(np.linspace(.1,np.pi/4,50), size = num_cells_per_type),
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
    synth_df = pl.DataFrame({"spike_times": [0] * len(all_spikes), # Not used but kept to match production data
                            "spike_clusters": all_clu_ID,
                            "cluster_group": all_clu_label,
                            "aligned_spike_times": all_spikes,
                            "aligned_spike_times_in_samples": [0] * len(all_spikes), # Not used
                            "spike_aligned_to_frame": np.around((all_spikes * 40))})
    
    synth_df = synth_df.with_columns(synth_df["spike_aligned_to_frame"].cast(pl.Float64))
    
    return synth_df

def return_spike_times_locked_to_behavioural_direction(behavioural_direction, 
                                                       number_of_spikes,
                                                       tuned_direction, 
                                                       dir_std = .1,
                                                       fps = 40, 
                                                       mean = 0, 
                                                       std_dev = 1,
                                                       add_noise = False,
                                                       noise_scale=0.1) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Selecting the behavioural direction of interest, this function will gather the samples
    of when that direction is occuring, convert into time and then generate spikes around that time.
    
    Inputs:
    + behavioural_direction: this is the tracking data that has been filtered on either hsa, hdir etc
    + tuned_drection: this is the set of angles to generate spikes for
    
    Returns: (both flat so they can unfold into dataframe)
    + flat list of spike times
    + flat list of cluster ids
    """
    all_spike_times = []
    all_cluster_ids = []
    
    for idx, items in enumerate(zip(tuned_direction, number_of_spikes, dir_std)):
        
        # Extract zipped contents 
        tuned_direction = items[0]
        spike_num = items[1]
        dir_std = items[2]
        
        # Extract behavioural ranges and generate spikes
        indices = np.where(np.logical_and((behavioural_direction >= (tuned_direction - dir_std)),(behavioural_direction <= (tuned_direction + dir_std))))[0]
        times = indices / fps
        spike_times = generate_spikes(spike_num, times, mean, std_dev)
    
        if add_noise:
                noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=len(spike_times))
                spike_times += noise
                
        all_spike_times.extend(spike_times)
        all_cluster_ids.extend([idx] * len(spike_times))
    
    return np.array(all_spike_times), np.array(all_cluster_ids)

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