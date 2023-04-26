import matplotlib.patches as mpatches
from databank import experiments_objects
from behave_analysis.process.process import Process
import random
import dill as pickle
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

def load_tracking_data():
    """Just for first session, load the tracking data."""
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        break
    file = os.path.join(session.file_path, "fully_processed_tracking_data.pickle")
    with open(file, "rb") as dill_file:
        tracking = pickle.load(dill_file)
    return tracking

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

def return_spike_times_locked_to_behavioural_direction(behavioural_direction, 
                                                       number_of_spikes,
                                                       direction_ranges, 
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
    
    for idx, direction_range in enumerate(direction_ranges, start=1):
    
        indices = np.where((direction_range[0] <= behavioural_direction) & (behavioural_direction <= direction_range[-1]))[0]
        times = indices / fps
        spike_times = generate_spikes(number_of_spikes, times, mean, std_dev)
    
        if add_noise:
                noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=len(spike_times))
                spike_times += noise
                
        all_spike_times.extend(spike_times)
        all_cluster_ids.extend([idx] * len(spike_times))
    
    all_spike_times = np.array(all_spike_times)
    all_cluster_ids = np.array(all_cluster_ids)
                
    return all_spike_times, all_cluster_ids

if __name__ == "__main__":
    
    # =================== LOAD DATA ================================
    np.random.seed(42)  # For reproducibility, you can remove this line for true randomnes
    tracking = load_tracking_data()
    headDirection = tracking["hdir"]  # head direction for each frame
    shelterDirection = tracking["hdir_shelt"]  # shelter direction for each frame
    
    print("loaded data")
    
    # Convert spike times to indices
    spike_times_hdir, cluster_ids = return_spike_times_locked_to_behavioural_direction(headDirection, 
                                          number_of_spikes = 100000, 
                                          direction_ranges = [(-0.1, 0.1), (2, 2.1), (-2.1, -2), (-1.7, -1.6)], 
                                          fps = 40,
                                          mean = 0, 
                                          std_dev = 1,
                                          add_noise= False,
                                          noise_scale = 25)

    # How many columns and rows should the plot have
    num_cols = 2
    num_rows = len(np.unique(cluster_ids))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 2 * num_rows), subplot_kw={'projection': None})
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    plot_counter = 0
    unique_cluster_ids = np.unique(cluster_ids)

    for idx in range(num_rows):
        for j in range(0, num_cols, 2):
            if plot_counter < len(unique_cluster_ids):
                # Get spike times for the current neuron (cluster)
                neuron_spike_times = spike_times_hdir[cluster_ids == unique_cluster_ids[plot_counter]]

                # Convert spike times to indices
                spike_indices = np.round(neuron_spike_times * 40).astype(int)

                # Retrieve head directions corresponding to spike indices
                spike_head_directions = headDirection[spike_indices]

                # Create a proxy artist for the legend
                blue_patch = mpatches.Patch(color="dodgerblue", alpha=0.75, label="Firing Rate")

                # Head Direction Tuning Curve
                sns.histplot(spike_head_directions, bins=100, kde=True, ax=axes[idx, j])
                axes[idx, j].set_xlabel("Head Direction (Radians)")
                axes[idx, j].set_ylabel("Spike Count")
                axes[idx, j].set_title(f"Head Direction Tuning Curve for Neuron {unique_cluster_ids[plot_counter]}")

                # Neuron Firing Polar Plot
                ax2 = axes[idx, j+1] = plt.subplot(num_rows, num_cols, (idx * num_cols) + j + 2, projection='polar')
                _, _, _ = ax2.hist(spike_head_directions, bins=36, alpha=0.75, color="dodgerblue")
                ax2.set_theta_zero_location("N")
                ax2.set_theta_direction(-1)
                ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
                ax2.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
                ax2.set_title(f"Neuron Firing Polar Plot for Neuron {unique_cluster_ids[plot_counter]}")

                plot_counter += 1

    plt.tight_layout()
    plt.show()  