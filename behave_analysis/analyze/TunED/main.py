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
        self.stimulus_idx, self.stimulus_bin_edges = self.compute_stimulus_indx()
        
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
        return np.digitize(self.stimulus_variable, stimulus_bin_edges) - 1, stimulus_bin_edges
    
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
    
    # @staticmethod
    # def compute_joint_prob(stimulusV1, stimulusV2, stimulusV2edges, stimulusV1edges):
    #     Pv2v1, _, _ = np.histogram2d(x = stimulusV1[0, :], # NOTE this is a hack to get the 1D array out of the 2D array
    #                                  y = stimulusV2[0, :], 
    #                                  normed = True)
    #     return Pv2v1
    
    # @staticmethod
    # def marginalize(joint_prob):
    #     raise NotImplementedError
    
    @staticmethod
    def tuning_function_nh2(tf_y, tf_y_s2, n_x, Py_x):
        """
        Also estimates SEM (new output).
        """
        Ncells, Nbins_x = tf_y.shape[0], Py_x.shape[1]
        tf_x_nh = np.zeros((Ncells, Nbins_x))
        tf_x_nh_sem = np.zeros((Ncells, Nbins_x))

        for c in range(Ncells):
            tf_x_nh[c, :] = np.sum((tf_y[c, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            s2_nh = np.sum((tf_y_s2[c, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            v_nh = s2_nh - tf_x_nh[c, :]**2
            tf_x_nh_sem[c, :] = np.sqrt(v_nh / n_x[c, :])
            
        return tf_x_nh, tf_x_nh_sem
    
if __name__ == '__main__':

    # Generate some data
    Ncells = 5
    Nsamples = 100
    Nbins = 10
    spike_count_matrix = np.random.uniform(10, 40, (Ncells, Nsamples)) # Generate random spike counts between 10 and 40 hertz
    
    # Gen the stimulus variables
    stimulusV1 = generate_stimulus_variable()
    stimulusV2 = generate_stimulus_variable()
    
    # Compute the tuning functions
    v1 = TunED(spike_count_matrix, stimulusV1, Nbins)
    tfv1, tf_semv1, tf_s2v1, nv1 = v1.tuning_function()
    v2 = TunED(spike_count_matrix, stimulusV2, Nbins)
    tfv2, tf_semv2, tf_s2v2, nv2 = v2.tuning_function()

    # Calculate the joint probability between the two stimulus variables
    Pv2v1, _, _ = np.histogram2d(stimulusV1[0, :], stimulusV2[0, :], normed = True)
    
    # Marginalize the joint probability
    Pv1 = np.sum(Pv2v1, axis=0)
    Pv2 = np.sum(Pv2v1, axis=1)
    
    # P(v2|v1)
    Pv2_v1 = Pv2v1 / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1)

    # P(v1|v2)
    Pv1_v2 = Pv2v1.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2)
    
    # Tuning function for v1 conditioned on v2
    tf_x_nh12, tf_x_nh_sem12 = TunED.tuning_function_nh2(tfv2, tf_s2v2, nv1, Pv2_v1)
    
    # Tuning function for v2 conditioned on v1
    tf_x_nh21, tf_x_nh_sem21 = TunED.tuning_function_nh2(tfv1, tf_s2v1, nv2, Pv1_v2)
    
    # Plot the tuning functions -------------------------------------------------------------
    # Create a figure
    fig, axs = plt.subplots(2, figsize=(10,10))
    
    # Tuning curves for v1
    for i in range(1):
        axs[0].errorbar(nv1[i], tfv1[i], tf_semv1[i], marker='.', linestyle='-')
        axs[0].errorbar(nv1[i], tf_x_nh12[i], tf_x_nh_sem12[i], linestyle='--')
    
    axs[0].set_title('Tuning to V1')
    axs[0].set_xlabel('v1')
    axs[0].set_ylabel('fr')
    axs[0].legend(['Tuning to v1','Tuning to v1 given NH that driver is v2'])

    # Tuning curves for v2
    for i in range(1):
        axs[1].errorbar(nv2[i], tfv2[i], tf_semv2[i], marker='.', linestyle='-')
        axs[1].errorbar(nv2[i], tf_x_nh21[i], tf_x_nh_sem21[i], linestyle='--')
    axs[1].set_title('Tuning to V2')
    axs[1].set_xlabel('v2')
    axs[1].set_ylabel('fr')
    
    plt.show()