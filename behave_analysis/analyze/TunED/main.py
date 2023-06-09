# Imports
import numpy as np
import matplotlib.pyplot as plt

# Utility function for testing NOTE can be deleted later

def generate_stimulus_variable():
    """
    Generate a random stimulus variable between 0 and 360 degrees for of length Nsamples
    """
    return np.random.uniform(0, 360, (1, Nsamples))

class TunED:
    """
    Ingests a spike count matrix and stimulus variable and performs TunED analysis
    """
    def __init__(self, spike_count_matrix, stimulus_variable, Nbins):
        self.spike_count_matrix = spike_count_matrix
        self.stimulus_variable = stimulus_variable
        self.Nbins = Nbins
        self.number_of_cells = spike_count_matrix.shape[0]
        self.number_of_samples = spike_count_matrix.shape[1]
        
        # Function calls
        self.__conduct_input_validation()
        self.stimulus_idx = self.compute_stimulus_indx()
        
    def __conduct_input_validation(self):
        """
        A private method to validate the input data
        """
        if Nsamples != self.stimulus_variable.shape[1]:
            raise ValueError('Mismatched input')
        if self.stimulus_variable.shape[0] != 1:
            raise ValueError('Unexpected input format')
    
    def compute_stimulus_indx(self):
        """
        A method to compute the stimulus index after binning
        NOTE binning might have to be done identically for each variable
        """
        stimulus_bin_edges =  np.linspace(np.min(self.stimulus_variable), np.max(self.stimulus_variable), Nbins+1)
        return np.digitize(self.stimulus_variable, stimulus_bin_edges) - 1
    
    def tuning_function(self):
        tf = np.zeros((Ncells, Nbins))
        tf_sem = np.zeros((Ncells, Nbins))
        tf_s2 = np.zeros((Ncells, Nbins))
        n = np.zeros((Ncells, Nbins))

        for cluster_idx in range(Ncells):
            for bin_idx in range(Nbins):
                mask = self.stimulus_idx == bin_idx # Boolean array of size (1, Nsamples). True when bin_index == stimulus_idx
                mask = mask[0] # Convert to 1D array of size (Nsamples,)
                
                # If there are any samples in this bin, compute the mean, variance, and standard error of the mean
                if np.sum(mask) > 0:
                    tf[cluster_idx, bin_idx] = np.mean(spike_count_matrix[cluster_idx, mask]) # For a given cluster and stimulus bin, extract all the spike counts and compute the mean spike count
                    tf_s2[cluster_idx, bin_idx] = np.var(spike_count_matrix[cluster_idx, mask]) + tf[cluster_idx, bin_idx]**2
                    tf_sem[cluster_idx, bin_idx] = np.std(spike_count_matrix[cluster_idx, mask]) / np.sqrt(np.sum(mask))
                    n[cluster_idx, bin_idx] = np.sum(mask)
                
                # Else if there are no samples in this bin, set the tuning function to -1
                else:
                    tf[cluster_idx, bin_idx] = -1
                    tf_s2[cluster_idx, bin_idx] = 0
                    tf_sem[cluster_idx, bin_idx] = 0
                    n[cluster_idx, bin_idx] = 0
        
        return tf, tf_sem, tf_s2, n
    
    @staticmethod
    def compute_joint_prob(stimulusV1, stimulusV2, stimulusV2edges, stimulusV1edges):
        Pv2v1, _, _ = np.histogram2d(x = stimulusV1, 
                                               y = stimulusV2, 
                                               range = [stimulusV2edges, stimulusV1edges], 
                                               normed=True)
        return Pv2v1
    
if __name__ == '__main__':

    # Generate some data
    Ncells = 5
    Nsamples = 100
    Nbins = 10
    spike_count_matrix = np.random.uniform(10, 40, (Ncells, Nsamples)) # Generate random spike counts between 10 and 40 hertz
    
    stimulusV1 = generate_stimulus_variable()
    stimulusV2 = generate_stimulus_variable()
    
    v1 = TunED(spike_count_matrix, stimulusV1, Nbins)
    tfv1, tf_semv1, tf_s2v1, nv1 = v1.tuning_function()
    v2 = TunED(spike_count_matrix, stimulusV2, Nbins)
    tfv2, tf_semv2, tf_s2v2, nv2 = v2.tuning_function()

    