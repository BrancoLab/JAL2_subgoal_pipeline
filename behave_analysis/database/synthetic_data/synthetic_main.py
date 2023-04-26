# Created dataframe has unused columns but required due to indexing in visualize efizz.
# TODO: remove unused columns from dataframe when indexing is fixed in efizz to be polars 

from behave_analysis.database.synthetic_data.generate_vectorial_synthetic_neurons import generate_synth_vectorial_cells
from behave_analysis.database.synthetic_data.poison_process_functions import ImhomogeneousProcess
from databank import experiments_objects
from behave_analysis.process.process import Process
from dataclasses import dataclass
import polars as pl
import random
import pickle as pkl
import numpy as np 

# Collect session data currently in the databank
for session_ID in experiments_objects:
    session = Process(session_ID).load_session()
    break

class GenerateFakeDataForRasters:
    def __init__(self):
        self.length_of_session = len(session.ttl.bonsai_TTL)
        self.sampling_rate = 30000
        self.onsets = self.generate_onsets(number_of_onsets = 10)
        self.spikes = self.generate_fake_spikes()
        self.clusters = self.generate_fake_clusters(number_of_spikes=len(self.spikes), number_of_clusters=10)
        self.dataframe = self.produce_polars_dataframe()

    def generate_onsets(self, number_of_onsets) -> list:
        """How many onsets do you want to generate? In essence how many
        trials do you want to simulate?

        Args:
            number_of_onsets (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            list: _description_
        """

        # Check if the number of onsets requested is within a valid range
        if number_of_onsets < 1 or number_of_onsets > self.length_of_session:
            raise ValueError("Count must be between 1 and {}".format(self.length_of_session))

        # Use the random.sample function to generate unique random integers
        return random.sample(range(self.length_of_session + 1), number_of_onsets)

    def poisson_process(self, onset_time):
        # Params for the kernel
        T = 10
        width_of_kernel = 0.5
        peak_intensity_of_kernel = 50
        num_bins = T
        bin_duration = T / num_bins

        object = ImhomogeneousProcess(
            time_end=onset_time + 10,
            peak_time=onset_time,
            width=width_of_kernel,
            peak_intensity=peak_intensity_of_kernel,
            kernel="gaussian",
        )

        spike_times = object.events  # Tunned to the kernel function

        return spike_times

    def generate_fake_spikes(self) -> list:
        # Create a list of lists
        spikes = []
        for idx, onset in enumerate(self.onsets):
            spikes.append(self.poisson_process(onset / self.sampling_rate))
            print("Generate trial {} of {}".format(idx + 1, len(self.onsets)))

        flat_list = [item for sublist in spikes for item in sublist]

        return flat_list

    def generate_fake_clusters(self, number_of_spikes, number_of_clusters):
        """Create a fake cluster of spikes all cluster 0"""
        # return [0] * number_of_spikes
        return np.random.randint(6, 10, size=number_of_spikes, dtype=np.int64)

    def produce_polars_dataframe(self):
        # Create a Polars DataFrame
        df = pl.DataFrame(
            {
                "spike_times": [0] * len(self.spikes), # Not used
                "spike_clusters": self.clusters,
                "cluster_group": ["good"] * len(self.spikes),
                "aligned_spike_times": self.spikes,
            }
        )

        # Print the DataFrame
        return df

if __name__ == "__main__":
    
    # TODO - Enter where you want your synthetic data to be saved
    path = r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_dataframe.csv"
    
    # Generate vectorial cells 
    head_direction_spikes, head_direction_clusters = generate_synth_vectorial_cells(cell_type = "Head_Direction", add_noise = False, number_of_spikes_per_cluster = 100000)
    shelter_direction_spikes, shelter_direction_clusters = generate_synth_vectorial_cells(cell_type = "Shelter_Direction", add_noise = False, number_of_spikes_per_cluster = 100000)

    # Create a Polars DataFrame - TODO - Comment out if you want to use the vectorial cells
    head_direction_only = pl.DataFrame({"spike_times": [0] * len(head_direction_spikes), # Not used
                                        "spike_clusters": head_direction_clusters,
                                        "cluster_group": ["good"] * len(head_direction_spikes),
                                         "aligned_spike_times": head_direction_spikes})
    head_direction_only.write_csv(path)
    
    # shelter_direction_only = pl.DataFrame({"spike_times": [0] * len(shelter_direction_spikes), # Not used
    #                                        "spike_clusters": shelter_direction_clusters,
    #                                        "cluster_group": ["good"] * len(shelter_direction_spikes),
    #                                        "aligned_spike_times": shelter_direction_spikes})
    # shelter_direction_only.write_csv(path)