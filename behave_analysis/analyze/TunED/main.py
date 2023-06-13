# OS Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class TunED:
    """
    Estimate tuning functions for 2 stimulus variables, along with tuning functions under the null hypothesis (NH) that only one variable is the driver.
    A new instance of this class should be created for each stimulus variable.
    """
    def __init__(self, spike_count_matrix, stimulus_variable, Nbins):
        """
        Args:
            spike_count_matrix (np.array): Firing rate matrix of shape (Ncells, Nsamples)
            stimulus_variable (np.array): The stimulus variable of shape (1, Nsamples)
            Nbins (_type_): _description_
        """
        self.spike_count_matrix = spike_count_matrix
        self.stimulus_variable = stimulus_variable
        self.Nbins = Nbins
        self.number_of_cells = spike_count_matrix.shape[0]
        self.number_of_samples = spike_count_matrix.shape[1]
        
        self.__conduct_input_validation()
        self.stimulus_idx, self.stimulus_bin_edges = self.compute_stimulus_indx()
        
    def __conduct_input_validation(self):
        """
        A private method to validate the input data. Raises a ValueError if the input data is not as expected.
        
        Returns: Nothing
        """
        if Nsamples != self.stimulus_variable.shape[1]:
            raise ValueError('Mismatched input')
        if self.stimulus_variable.shape[0] != 1:
            raise ValueError('Unexpected input format')
    
    def compute_stimulus_indx(self):
        """
        A method to compute the stimulus index after binning. NOTE binning might have to be done identically
        for each variable. So first create bin edges for a given variable ending in stimulus_bin_edges
        of shape (Nbins, ). Then use digitize to discretize he stimulus variable into 
        
        Returns:
            stimulus_idx: <np.ndarray> of size (1, Nsamples). The stimulus index for each sample.
            stimulus_bin_edges: <np.ndarray> of size (Nbins, ). The bin edges used to bin the stimulus variable.
        """
        stimulus_bin_edges =  np.linspace(np.min(self.stimulus_variable), np.max(self.stimulus_variable), self.Nbins+1)
        stimulus_idx = np.digitize(self.stimulus_variable, stimulus_bin_edges) - 1
        return stimulus_idx, stimulus_bin_edges
    
    def tuning_function(self, Ncells, Nbins, spike_count_matrix):
        """The purpose of this function is to compute the tuning function for each cell which equates
        to the corresponding mean spike count for each stimulus bin.
        
        Inputs:
            Ncells: <int> The number of cells in the spike count matrix
            Nbins: <int> The number of bins to use to bin up the stimulus variable
            spike_count_matrix: <np.ndarray> of size (Ncells, Nsamples). The spike count matrix.

        Returns:
            tf: <np.ndarray> of size (Ncells, Nbins). The tuning function for each cell.
            tf_sem: <np.ndarray> of size (Ncells, Nbins). The standard error of the mean for each cell.
            tf_s2: <np.ndarray> of size (Ncells, Nbins). The variance of the tuning function for each cell.
            n: <np.ndarray> of size (Ncells, Nbins). The number of samples in each bin for each cell.
        """
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
                    
        bin_centres = (self.stimulus_bin_edges[1:] + self.stimulus_bin_edges[:-1]) / 2
        
        return tf, tf_sem, tf_s2, n, bin_centres
    
    @staticmethod
    def compute_joint_prob(stimulusV1, stimulusV2, stimulusV2edges, stimulusV1edges):
        """
        A static method to compute the joint probability between two stimulus variables. Returns a joint probability
        table of size (Nbins, Nbins) that can be used to compute the mutual information between the two variables.
        There is a unit test below to ensure that the joint probability table is computed correctly.
        
        Inputs:
            stimulusV1: <np.ndarray> of size (1, Nsamples). The first stimulus variable.
            stimulusV2: <np.ndarray> of size (1, Nsamples). The second stimulus variable.
            stimulusV2edges: <np.ndarray> of size (Nbins, ). The bin edges used to bin the second stimulus variable.
            stimulusV1edges: <np.ndarray> of size (Nbins, ). The bin edges used to bin the first stimulus variable.
        
        Returns:
            Pv2v1, _, _: Joint probability table of size (Nbins, Nbins)
        """
        Pv2v1, xedges, yedges = np.histogram2d(x = stimulusV1[0, :], # index to get the 1D array out of the 2D array
                                               y = stimulusV2[0, :], # index to get the 1D array out of the 2D array
                                               bins = Nbins,
                                               range = [[min(stimulusV1edges), max(stimulusV1edges)], 
                                                        [min(stimulusV2edges), max(stimulusV2edges)]],
                                               density = True)
        
        # ----------- Unit Test Logic below for internal consistency----------------
        
        # Calculate bin widths in both dimensions
        x_bin_width = xedges[1:] - xedges[:-1]
        y_bin_width = yedges[1:] - yedges[:-1]

        # Calculate bin areas by taking the outer product of the bin widths
        bin_areas = np.outer(y_bin_width, x_bin_width)

        # Multiply each bin value with its corresponding bin area and sum over all bins - Definition per
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
        total = np.sum(Pv2v1 * bin_areas)
        
        # Assertion Test for internal consistency
        assert np.isclose(total, 1), 'The joint probability does not sum to 1'
        assert np.all(Pv2v1 >= 0), 'The joint probability has negative values'
        assert np.all(Pv2v1 <= 1), 'The joint probability has values greater than 1'
        assert np.isclose(sum(np.sum(Pv2v1 * bin_areas , axis=0)), 1), 'The marginal probability of V1 does not sum to 1'
        assert np.isclose(sum(np.sum(Pv2v1 * bin_areas , axis=1)), 1), 'The marginal probability of V2 does not sum to 1'
        
        return Pv2v1, xedges, yedges
    
    @staticmethod
    def marginalize(joint_prob):
        """
        Computes the marginal probability distribution of a joint probability distribution.
        """
        marginal_x = np.sum(joint_prob, axis=0)
        marginal_y = np.sum(joint_prob, axis=1)
        
        return marginal_x, marginal_y
    
    @staticmethod
    def tuning_function_null_hypothesis_PYX(tf_y, tf_y_s2, n_x, Py_x):
        """
        Compute the tuning_function to 'x' expected from the Null hypothesis that its structure is entirely a 
        consequence of tuning to a quantity 'y' (which may be correlated with 'x'). In essence:
         --> P(x = firing curve|y)
         
        Inputs:
            tf_y: <np.ndarray> of size (Ncells, Nbins_y). The mean firing rate of each cell as a function of stimulus y.
            tf_y_s2: <np.ndarray> of size (Ncells, Nbins_y). The variance of the firing rate of each cell as a function of stimulus y.
            n_x: <np.ndarray> of size (Ncells, Nbins_x). The number of samples in each bin of stimulus x.
            Py_x: <np.ndarray> of size (Nbins_y, Nbins_x). The probability of stimulus y given stimulus x. P(y|x)
            
        Returns:
            tf_x_nh: <np.ndarray> of size (Ncells, Nbins_x). The tuning function to stimulus x expected from the null hypothesis.
            tf_x_nh_sem: <np.ndarray> of size (Ncells, Nbins_x). The standard error of the tuning function to stimulus x expected from the null hypothesis.
        """
        
        # Init
        Ncells, Nbins_x = tf_y.shape[0], Py_x.shape[1]
        tf_x_nh = np.zeros((Ncells, Nbins_x))
        tf_x_nh_sem = np.zeros((Ncells, Nbins_x))

        for cluster in range(Ncells):
            
            # In essence we are computing the expectation of the firing rate of each cell as a function of stimulus x
            # The value of the expectation is the sum of the probability of each stimulus y given stimulus x, times the
            # firing rate of each cell as a function of stimulus y.
            # Reshape tf_cluster into a column vector for matrix multiplication - This is the point of [:, None]
            tf_x_nh[cluster, :] = np.sum((tf_y[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            s2_nh = np.sum((tf_y_s2[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            v_nh = s2_nh - tf_x_nh[cluster, :]**2
            tf_x_nh_sem[cluster, :] = np.sqrt(v_nh / n_x[cluster, :])
            
        return tf_x_nh, tf_x_nh_sem
    
if __name__ == '__main__':

    # Set random seed and parameters
    np.random.seed(0)
    Ncells = 1
    Nsamples = 100000 # Number of samples to generate, i.e. number of frames
    Nbins = 10 # Number of bins to use to bin up the stimulus variable
    
    # Generate stimuli
    stimulusV1 = np.random.randn(1, Nsamples) # Driver stimulus
    stimulusV2 = stimulusV1 * 0.6 + np.random.randn(1, Nsamples) # Passenger stimulus
    
    # Print the stimulus variables as a correlation matrix, to show variable 2 is correlated with variable 1
    print(np.corrcoef(stimulusV1, stimulusV2))

    # Generate spike trains from a Poisson process with a rate that depends on the stimulus V1
    frate = 0.1*(stimulusV1[0, :] > 1.0) * stimulusV1[0, :]
    frate = np.minimum(1, frate)
    raster = np.zeros((1, Nsamples))
    raster[0, :] = poisson.rvs(frate)
    
    # Compute the tuning functions
    v1Object = TunED(raster, stimulusV1, Nbins)
    tfv1, tf_semv1, tf_s2v1, nv1, bin_centresV1 = v1Object.tuning_function(Ncells, Nbins, raster)
    v2Object = TunED(raster, stimulusV2, Nbins)
    tfv2, tf_semv2, tf_s2v2, nv2, bin_centresV2 = v2Object.tuning_function(Ncells, Nbins, raster)

    # Joint probability of the stimuli --------------------------------------------------------
    jointProb_stimuli, _, _ = TunED.compute_joint_prob(stimulusV1, 
                                                       stimulusV2, 
                                                       stimulusV2edges = v2Object.stimulus_bin_edges, 
                                                       stimulusV1edges = v1Object.stimulus_bin_edges)
        
    # Marginalize the joint probability ------------------------------------------------------
    Pv1, Pv2 = TunED.marginalize(jointProb_stimuli)
    
    # Compute the conditional probabilities for the stimulus ---------------------------------------------------
    Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
    # Tranpose to ensure broadcasting works correctly row wise instead of column wise
    Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
    
    # NULL Hypothesis tests ---------------------------------------------------------------------
    
    # apparent tuning to 'v2' given NH that cell is driven purely by v1 P(v2|v1):
    tf_x_nh2_driven_by_1, tf_x_nh_sem21 = TunED.tuning_function_null_hypothesis_PYX(tfv1, tf_s2v1, nv2, Pv1_v2)
    
    # apparent tuning to 'v1' given NH that cell is driven purely by v2 P(v1|v2):
    tf_x_nh1_drive_by_2, tf_x_nh_sem12 = TunED.tuning_function_null_hypothesis_PYX(tfv2, tf_s2v2, nv1, Pv2_v1)
    
    # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(23, 5))

    ax[0].plot(bin_centresV1, tfv1[0, :], '.-', label='Tuning to V1', color="cornflowerblue")
    ax[0].fill_between(bin_centresV1, tfv1[0, :] - tf_semv1[0, :], tfv1[0, :] + tf_semv1[0, :], alpha=0.1, color="cornflowerblue")
    ax[0].plot(bin_centresV1, tf_x_nh1_drive_by_2[0, :], '.--', label='Tuning to v1 given NH that driver is v2', color='darkorchid')
    ax[0].fill_between(bin_centresV1, tf_x_nh1_drive_by_2[0, :] - tf_x_nh_sem12[0, :], tf_x_nh1_drive_by_2[0, :] + tf_x_nh_sem12[0, :], alpha=0.1, color='darkorchid')
    ax[0].set_xlabel('v1')
    ax[0].set_ylabel('fr')
    ax[0].legend(loc='upper right')
    ax[0].set_title("Tuning to V1", fontweight="bold")

    ax[1].plot(bin_centresV2, tfv2[0, :], '.-', label='Tuning to V2', color='cornflowerblue')
    ax[1].fill_between(bin_centresV2, tfv2[0, :] - tf_semv2[0, :], tfv2[0, :] + tf_semv2[0, :], color='cornflowerblue', alpha=0.1)
    ax[1].plot(bin_centresV2, tf_x_nh2_driven_by_1[0, :], '.--', label='Tuning to v2 given NH that driver is v1', color='darkorchid')
    ax[1].fill_between(bin_centresV2, tf_x_nh2_driven_by_1[0, :] - tf_x_nh_sem21[0, :], tf_x_nh2_driven_by_1[0, :] + tf_x_nh_sem21[0, :], color='darkorchid', alpha=0.1)
    ax[1].set_xlabel('v2')
    ax[1].set_ylabel('fr')
    ax[1].legend(loc='upper right')
    ax[1].set_title("Tuning to V2", fontweight="bold")

    plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
    plt.show()
    
    
