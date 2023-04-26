from databank import experiments_objects
from behave_analysis.process.process import Process
import random
import dill as pickle
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
np.random.seed(42)  # For reproducibility, you can remove this line for true randomness

# =================== UTILITY FUNCTIONS ===================

def generate_spikes(spike_count, 
                    times, 
                    mean, 
                    std_dev) -> np.ndarray:
    """
    Produces a gaussian of spikes around times of behavioural relevant events.
    """
    valid_indices = np.arange(len(times))
    chosen_indices = np.random.choice(valid_indices, size=spike_count)
    gaussian_offsets = np.random.normal(mean, std_dev, size=spike_count)
    spike_times = times[chosen_indices] + gaussian_offsets
    spike_times.sort()
    return spike_times

def return_spike_times_locked_to_behavioural_direction(behavioural_direction, 
                                                       number_of_spikes, 
                                                       range = (-0.1, 0.1), 
                                                       fps = 40, 
                                                       mean = 0, 
                                                       std_dev = 1) -> np.ndarray:
    
    """
    Selecting the behavioural direction of interest, this function will gather the samples
    of when that direction is occuring, convert into time and then generate spikes around that time. 
    """
    indices = np.where((range[0] <= behavioural_direction) & (behavioural_direction <= range[-1]))[0]
    times = indices / fps
    spike_times = generate_spikes(number_of_spikes, times, mean, std_dev)
    return spike_times

# =================== LOAD DATA ================================

# Get Session data
for session_ID in experiments_objects:
    session = Process(session_ID).load_session()
    break

# Get tracking data
file = os.path.join(session.file_path, "fully_processed_tracking_data.pickle")
with open(file, "rb") as dill_file:
    tracking = pickle.load(dill_file)

# Extract the required information from the tracking data
headDirection = tracking["hdir"]  # head direction for each frame
shelterDirection = tracking["hdir_shelt"]  # shelter direction for each frame

# =================== CREATE SYNTHETIC DATA ===================
number_of_clusters = 10

result = np.array([])
shift = 0
clusters = []
for i in range(number_of_clusters): 
    
    spike_times_hdir = return_spike_times(headDirection, 
                                          number_of_spikes = 1000000, 
                                          range = (-0.1 + shift, 0.1 + shift), 
                                          fps = 40,
                                          mean = 0, 
                                          std_dev = 1)
    
    result = np.concatenate((result, spike_times_hdir))
    shift += 0.3
    clusters.append([i] * len(spike_times_hdir))

clusters = [item for sublist in clusters for item in sublist]

polar_df_hdir = pl.DataFrame({"spike_times": [0] * len(result),
                              "spike_clusters": clusters,
                              "cluster_group": ["good"] * len(result),
                              "aligned_spike_times": result})

# ------------Produce robot data for the shelter direction----------------
# spike_times_Sheldir = return_spike_times(shelterDirection, number_of_spikes = 1000000, range = (-0.1, 0.1), fps = 40, mean = 0, std_dev = 1)
# polar_df_sheldir = pl.DataFrame(
#     {
#         "spike_times": [0] * len(spike_times_Sheldir),
#         "spike_clusters": np.random.randint(20, 25, size=len(spike_times_Sheldir), dtype=np.int64),
#         "cluster_group": ["good"] * len(spike_times_Sheldir),
#         "aligned_spike_times": spike_times_Sheldir,
#     }
# )

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Plot the head direction tuning curve and the neuron firing polar plot
    import matplotlib.patches as mpatches

    # Create a proxy artist for the legend
    blue_patch = mpatches.Patch(color="dodgerblue", alpha=0.75, label="Firing Rate")

    # Convert spike times to indices
    spike_indices = np.round(spike_times_hdir * 40).astype(int)

    # Retrieve head directions corresponding to spike indices
    spike_head_directions = headDirection[spike_indices]

    # Generate combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Head Direction Tuning Curve
    sns.histplot(spike_head_directions, bins=100, kde=True, ax=ax1)
    ax1.set_xlabel("Head Direction (Radians)")
    ax1.set_ylabel("Spike Count")
    ax1.set_title("Head Direction Tuning Curve")

    # Neuron Firing Polar Plot
    ax2 = plt.subplot(1, 2, 2, polar=True)
    hist, _, _ = ax2.hist(spike_head_directions, bins=36, alpha=0.75, color="dodgerblue")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax2.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax2.set_title("Neuron Firing Polar Plot")
    ax2.legend(handles=[blue_patch], loc="upper left")

    plt.show()
