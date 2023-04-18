from dataclasses import dataclass
import polars as pl
import random
from behave_analysis.database.synthetic_data.synthetic_data_functions import ImhomogeneousProcess
import pickle as pkl

@dataclass(frozen=True)
class FakeData:
    """A class to store experiment by experiment information"""
    on_sets: list # Stimulus onset samples
    dataFrame: pl.DataFrame # Dataframe of the data
    
class GenerateFakeData:
    def __init__(self):
        self.length_of_session = 225336000
        self.sampling_rate = 30000
        self.onsets = self.generate_onsets(number_of_onsets = 5)
        self.spikes = self.generate_fake_spikes()
        self.clusters = self.generate_fake_clusters(number_of_spikes = len(self.spikes), number_of_clusters = 10)
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
        
        object = ImhomogeneousProcess(time_end = onset_time + 10, 
                                      peak_time = onset_time, 
                                      width = width_of_kernel, 
                                      peak_intensity = peak_intensity_of_kernel,
                                      kernel = "gaussian")
        
        spike_times = object.events # Tunned to the kernel function
        
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
        return [random.randint(0, number_of_clusters) for _ in range(number_of_spikes)]

    def produce_polars_dataframe(self):
        
        # Create a Polars DataFrame
        df = pl.DataFrame(
            {
                "spike_clusters": self.clusters,
                "aligned_spike_times": self.spikes,
                "cluster_group": ["good"] * len(self.spikes),
            }
        )

        # Print the DataFrame
        return df

if __name__ == "__main__":

    path = r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_dataframe.csv"
    generator = GenerateFakeData()
    dataframe = generator.dataframe
    onsets = generator.onsets
    
    with open(r"C:\Users\laurence\Documents\JAL-pipeline\behave_analysis\database\synthetic_data\synthetic_onsets.pkl", "wb") as f:
        pkl.dump(onsets, f)
    
    print(dataframe)
    dataframe.write_csv(path)
    

        



