from databank import experiments_objects
from behave_analysis.process.process import Process
import random
import dill as pickle
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

for session_ID in experiments_objects:
    session = Process(session_ID).load_session()
    break

file = os.path.join(session.file_path, "fully_processed_tracking_data.pickle")
with open(file, "rb") as dill_file:
    tracking = pickle.load(dill_file)

headDirection = tracking["hdir"]  # head direction for each frame

print(headDirection)

fps = 40

# Filter head_directions between -0.1 and 0.1 radians
indices = np.where((-0.1 <= headDirection) & (headDirection <= 0.1))[0]

# Calculate the time of each frame
times = indices / fps

# Generate the desired number of spikes
def generate_spikes(spike_count, times, mean, std_dev):
    np.random.seed(42)  # For reproducibility, you can remove this line for true randomness
    valid_indices = np.arange(len(times))
    chosen_indices = np.random.choice(valid_indices, size=spike_count)
    gaussian_offsets = np.random.normal(mean, std_dev, size=spike_count)
    spike_times = times[chosen_indices] + gaussian_offsets
    spike_times.sort()
    return spike_times

# Example usage
spike_count = 100000  # The desired number of spikes
mean = 0  # Mean offset for the Gaussian distribution
std_dev = 0.1  # Standard deviation for the Gaussian distribution
spike_times = generate_spikes(spike_count, times, mean, std_dev)

# clusters = np.random.randint(1, 11, size=len(spike_times))

# Create a polar dataframe
polar_df_hdir = pl.DataFrame(
    {
        #  "spike_clusters": clusters.astype(np.int64),
        "spike_times": [0] * len(spike_times),
        "spike_clusters": np.random.randint(0, 5, size=len(spike_times), dtype=np.int64),
        "cluster_group": ["good"] * len(spike_times),
        "aligned_spike_times": spike_times,
    }
)

# # Display the polar dataframe
print(polar_df_hdir)

if __name__ == "__main__":
    
    # Plot the head direction tuning curve and the neuron firing polar plot
    import matplotlib.patches as mpatches

    # Create a proxy artist for the legend
    blue_patch = mpatches.Patch(color="dodgerblue", alpha=0.75, label="Firing Rate")

    # Convert spike times to indices
    spike_indices = np.round(spike_times * fps).astype(int)

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
