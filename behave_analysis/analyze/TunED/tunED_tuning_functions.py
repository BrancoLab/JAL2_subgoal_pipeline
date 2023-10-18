# Import OS Libaries

import numpy as np
from loguru import logger

class ComputeObservedTuningFunction:
    """
    The main purpose of this class is to compute the tuning function for each cell which equates
    to the corresponding mean spike count for each stimulus type (hdir, hsa etc).
        
    Inputs to the main function compute_tuning_function are:
        Ncells: <int> The number of cells in the spike count matrix
        Nbins: <int> The number of bins to use to bin up the stimulus variable
        spike_count_matrix: <np.ndarray> of size (Ncells, Nsamples). The spike count matrix.

    Returns:
        tf: <np.ndarray> of size (Ncells, Nbins). The tuning function for each cell.
        tf_sem: <np.ndarray> of size (Ncells, Nbins). The standard error of the mean for each cell.
        tf_s2: <np.ndarray> of size (Ncells, Nbins). The variance of the tuning function for each cell. - Second moment
        n: <np.ndarray> of size (Ncells, Nbins). The number of samples in each bin for each cell.
        sd: <np.ndarray> of size (Ncells, Nbins). The standard deviation of the tuning function for each cell.
        
    #NOTE Currently cell number is hardcoded to be 1 - remove loop over cells or make it work for multiple cells
    """
    
    def __init__(self, spike_count_matrix: np.ndarray, stimulus_variable: np.ndarray, Nbins: int, Nsamples: int):
        self.behavioural_angle = stimulus_variable
        self.length_of_session = Nsamples
        self.Nbins = Nbins
        self.Ncells = 1
        self.spike_count_matrix = spike_count_matrix
        self.__conduct_input_validation()
        self.stimulus_idx, self.stimulus_bin_edges = self.compute_stimulus_indx()
        self.tuning_func, self.tuning_func_sem, self.tuning_func_s2, self.n, self.bin_centres, self.tuning_func_sd, self.skipCluster = self.compute_tuning_function()

    def __conduct_input_validation(self) -> None:
        """
        A private method to validate the input data. Raises a ValueError if the input data is not as expected.
        """
        
        if self.length_of_session != self.behavioural_angle.shape[1]:
            raise ValueError('Mismatched input')
        if self.behavioural_angle.shape[0] != 1:
            raise ValueError('Unexpected input format')
        if self.length_of_session == 0:
            raise ValueError('No samples passed to the function, length of session is 0')

    def compute_stimulus_indx(self):
        """
        Assign a bin index to each stimulus sample.
        """
        stimulus_bin_edges =  np.linspace(np.min(self.behavioural_angle),
                                          np.max(self.behavioural_angle) + np.finfo(float).eps, 
                                          self.Nbins + 1) # Add one to the number of bins to get the number of bin edges
        stimulus_idx = np.digitize(x = self.behavioural_angle, bins = stimulus_bin_edges) - 1 # Subtract 1 to make the bin index start at 0
        
        # Unit tests
        assert len(stimulus_idx[0]) == self.length_of_session, 'The length of the stimulus index does not match the length of the session'
        assert np.all(stimulus_idx[0] >= 0), "Some indices in stimulus_idx are less than 0."
        assert np.all((stimulus_idx >= 0) & (stimulus_idx <= self.Nbins)), "stimulus_idx contains indices outside the range [0, Nbins]."

        return stimulus_idx, stimulus_bin_edges

    def compute_tuning_function(self) -> tuple:
        """
        The main purpose of this class is to compute this tuning function for each cell which equates to the corresponding mean spike count for each stimulus type (hdir, hsa etc).
        
        Inputs to the main function compute_tuning_function are:
            Ncells: <int> The number of cells in the spike count matrix
            Nbins: <int> The number of bins to use to bin up the stimulus variable
            spike_count_matrix: <np.ndarray> of size (Ncells, Nsamples). The spike count matrix.

        Returns:
            tf: <np.ndarray> of size (Ncells, Nbins). The tuning function for each cell.
            tf_sem: <np.ndarray> of size (Ncells, Nbins). The standard error of the mean for each cell.
            tf_s2: <np.ndarray> of size (Ncells, Nbins). The variance of the tuning function for each cell. - Second moment
            n: <np.ndarray> of size (Ncells, Nbins). The number of samples in each bin for each cell.
            sd: <np.ndarray> of size (Ncells, Nbins). The standard deviation of the tuning function for each cell.
            
        # TODO: Currently this function is only implemented for a single cell. Need to implement for multiple cells.
        """

        # Initialize the output arrays
        tf = np.zeros((self.Ncells, self.Nbins))
        tf_sem = np.zeros((self.Ncells, self.Nbins))
        tf_s2 = np.zeros((self.Ncells, self.Nbins))
        n = np.zeros((self.Ncells, self.Nbins))
        std_dev = np.zeros((self.Ncells, self.Nbins))
        noSamplesInOneBin = False # If one of the bins has no samples it could create issues somewhere else in the maths so for now exclude this cluster from the analysis
        
        # Loop over each cluster and each stimulus bin
        for cluster_idx in range(self.Ncells):
            for bin_idx in range(self.Nbins):
                mask = self.stimulus_idx == bin_idx # Boolean array of size (1, Nsamples). True when bin_index == stimulus_idx
                mask = mask[0] # Convert to 1D array of size (Nsamples,)
                
                # If there are any samples in this bin, compute the mean, variance, and standard error of the mean
                if np.sum(mask) > 0:
                    tf[cluster_idx, bin_idx] = np.mean(self.spike_count_matrix[cluster_idx, mask]) # For a given cluster and stimulus bin, extract all the spike counts and compute the mean spike count
                    tf_s2[cluster_idx, bin_idx] = np.var(self.spike_count_matrix[cluster_idx, mask]) + tf[cluster_idx, bin_idx]**2
                    tf_sem[cluster_idx, bin_idx] = np.std(self.spike_count_matrix[cluster_idx, mask]) / np.sqrt(np.sum(mask))
                    n[cluster_idx, bin_idx] = np.sum(mask) # how many samples are in this bin
                    std_dev[cluster_idx, bin_idx] = np.std(self.spike_count_matrix[cluster_idx, mask])
                
                # Else if there are no samples in this bin, set the tuning function to -1
                else:
                    logger.error(f"Cluster {cluster_idx} has no samples in bin {bin_idx} - Cluster 0 is a incorrect index, this is a placeholder for now, need to fix this")
                    tf[cluster_idx, bin_idx] = -1
                    tf_s2[cluster_idx, bin_idx] = 0
                    tf_sem[cluster_idx, bin_idx] = 0
                    n[cluster_idx, bin_idx] = 0
                    std_dev[cluster_idx, bin_idx] = 0
                    noSamplesInOneBin = True
                    break # Break out of the loop over stimulus bins

        bin_centres = (self.stimulus_bin_edges[1:] + self.stimulus_bin_edges[:-1]) / 2
        
        return tf, tf_sem, tf_s2, n, bin_centres, std_dev, noSamplesInOneBin

class ComputeNullHypothesisTuningFunction:
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
    
    def __init__(self, observed_tuning_function, observed_tuning_function_s2, num_values_for_Px, conditional_Py_x):
        self.Ncells = 1
        self.tuning_func_nh, self.tuning_func_nh_sem, self.tuning_func_nh_sd = self.compute_tuning_function(observed_tuning_function, 
                                                                                    observed_tuning_function_s2, 
                                                                                    num_values_for_Px, 
                                                                                    conditional_Py_x)

    def compute_tuning_function(self, tf_y, tf_y_s2, n_x, Py_x):
        """
        This function computes the NH. Which is the expected tuning function of y given x.
        Such that:
        --> NH for X = E[tf_Y|X] = ∫dy tf_y P(y|x)
        
        Steps:
            1. Convert it into a column vector (i.e., a matrix with one column)
            2. ∫dy tf_y P(y|x) - this is the expected value of the tuning function of y given x E[tf_y|x]
        
        Inputs:
            Py_x: <np.ndarray> of size (Nbins, Nbins). The probability of stimulus y given stimulus x. P(y|x)
            ty_y: <np.ndarray> of size (Ncells, Nbins). The mean firing rate of each cell as a function of stimulus y.
            
            Outputs:
            tf_x_nh: <np.ndarray> of size (Ncells, Nbins). The tuning function to stimulus x expected from the null hypothesis.
        """
        
        assert Py_x.shape[0] == Py_x.shape[1], "Py_x must be square" # Verify that this has to hold true and this isn't just tired logic 
        
        Ncells, Nbins_x = tf_y.shape[0], Py_x.shape[1] # Given Py_x is square, Nbins_y = Nbins_x, it doesn't matter which we use for the inner product below
        
        # Initialize the output arrays
        tf_x_nh = np.zeros((self.Ncells, Nbins_x))
        tf_x_nh_sem = np.zeros((self.Ncells, Nbins_x))
        tf_x_nh_sd = np.zeros((self.Ncells, Nbins_x))
        
        for cluster in range(self.Ncells):
            tf_x_nh[cluster, :] = np.sum((tf_y[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0) # Turn ty_y into a column vector and multiply by Py_x
            s2_nh = np.sum((tf_y_s2[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            v_nh = s2_nh - tf_x_nh[cluster, :]**2
            tf_x_nh_sem[cluster, :] = np.sqrt(v_nh / n_x[cluster, :])
            tf_std_dev = np.sqrt(v_nh)
        return tf_x_nh, tf_x_nh_sem, tf_std_dev