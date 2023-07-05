"""
Lingo:
+ Stimulus means the stimulus variable (e.g. hdir, hsa or other behavioural variables) that are simultaneously recorded with the spikes

TODO:
+ There should be some quality checks done on the ingested data because I found a spike count at 130  in one frame for one cell which is impossible so data quality is not there yet.
"""

# OS Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from loguru import logger
from abc import abstractmethod
from scipy.stats import norm, poisson, binom
import pickle 

def create_global_bin_edges(v1, v2, Nbins):
    """
    Currently not used
    """
    v1min = np.min(v1)
    v1max = np.max(v1)
    
    v2min = np.min(v2)
    v2max = np.max(v2)
    
    totmin = np.min([v1min, v2min])
    totmax = np.max([v1max, v2max])
    
    stimulus_bin_edges =  np.linspace(np.min(self.behavioural_angle), np.max(self.behavioural_angle) + np.finfo(float).eps, Nbins + 1) # Add one to the number of bins to get the number of bin edges
    
    return stimulus_bin_edges

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
    """
    def __init__(self, spike_count_matrix: np.ndarray, stimulus_variable: np.ndarray, Nbins: int, Nsamples: int):
        self.behavioural_angle = stimulus_variable
        self.length_of_session = Nsamples
        self.Nbins = Nbins
        self.Ncells = 1
        self.spike_count_matrix = spike_count_matrix
        self.__conduct_input_validation()
        self.stimulus_idx, self.stimulus_bin_edges = self.compute_stimulus_indx()
        self.tuning_func, self.tuning_func_sem, self.tuning_func_s2, self.n, self.bin_centres, self.tuning_func_sd = self.compute_tuning_function()

    def __conduct_input_validation(self) -> None:
        """
        A private method to validate the input data. Raises a ValueError if the input data is not as expected.
        """
        if self.length_of_session != self.behavioural_angle.shape[1]:
            raise ValueError('Mismatched input')
        if self.behavioural_angle.shape[0] != 1:
            raise ValueError('Unexpected input format')

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
                    n[cluster_idx, bin_idx] = np.sum(mask)
                    std_dev[cluster_idx, bin_idx] = np.std(self.spike_count_matrix[cluster_idx, mask])
                
                # Else if there are no samples in this bin, set the tuning function to -1
                else:
                    logger.error(f"Cluster {cluster_idx} has no samples in bin {bin_idx}")
                    tf[cluster_idx, bin_idx] = -1
                    tf_s2[cluster_idx, bin_idx] = 0
                    tf_sem[cluster_idx, bin_idx] = 0
                    n[cluster_idx, bin_idx] = 0
                    std_dev[cluster_idx, bin_idx] = 0

        bin_centres = (self.stimulus_bin_edges[1:] + self.stimulus_bin_edges[:-1]) / 2
        
        # Adding limit logic to see if thats why the tuning curves dont match darios
        
        # Compute limits from input data
        # limits = np.std(np.vstack((self.behavioural_angle,)), axis=1) * np.array([-3, 3])

        # # Find indices where the bin centers are within the limits
        # good_indices = (bin_centres > limits[0]) & (bin_centres < limits[1])

        # # Use these indices to select only the valid entries in each array
        # tf = tf[:, good_indices]
        # tf_s2 = tf_s2[:, good_indices]
        # tf_sem = tf_sem[:, good_indices]
        # n = n[:, good_indices]
        # bin_centres = bin_centres[good_indices]
        # std_dev = std_dev[:, good_indices]

        return tf, tf_sem, tf_s2, n, bin_centres, std_dev

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
        self.tuning_func_nh, self.tuning_func_nh_sem = self.compute_tuning_function(observed_tuning_function, 
                                                                                    observed_tuning_function_s2, 
                                                                                    num_values_for_Px, 
                                                                                    conditional_Py_x)

    def compute_tuning_function(self, tf_y, tf_y_s2, n_x, Py_x):
        """
        What is the mean firing rate conditioned on another variable?
        Steps:
            1. Convert it into a column vector (i.e., a matrix with one column)
        
        Inputs:
            Py_x: <np.ndarray> of size (Nbins, Nbins). The probability of stimulus y given stimulus x. P(y|x)
            ty_y: <np.ndarray> of size (Ncells, Nbins). The mean firing rate of each cell as a function of stimulus y.
            
            Outputs:
            tf_x_nh: <np.ndarray> of size (Ncells, Nbins). The tuning function to stimulus x expected from the null hypothesis.
        """
        assert Py_x.shape[0] == Py_x.shape[1], "Py_x must be square" # Verify that this has to hold true and this isn't just tired logic 
        
        Ncells, Nbins_x = tf_y.shape[0], Py_x.shape[1] # Given Py_x is square, Nbins_y = Nbins_x, it doesn't matter which we use for the inner product below
        tf_x_nh = np.zeros((self.Ncells, Nbins_x))
        tf_x_nh_sem = np.zeros((self.Ncells, Nbins_x))
        for cluster in range(self.Ncells):
            tf_x_nh[cluster, :] = np.sum((tf_y[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0) # Turn ty_y into a column vector and multiply by Py_x
            s2_nh = np.sum((tf_y_s2[cluster, :][:, None] @ np.ones((1, Nbins_x))) * Py_x, axis=0)
            v_nh = s2_nh - tf_x_nh[cluster, :]**2
            tf_x_nh_sem[cluster, :] = np.sqrt(v_nh / n_x[cluster, :])
        return tf_x_nh, tf_x_nh_sem
    
class TunEDModelStats:    
    @staticmethod
    def compute_joint_prob(stimulusV1, stimulusV2, stimulusV2edges, stimulusV1edges, Nbins):
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
            Pv1v2, _, _: Joint probability table of size (Nbins, Nbins) the x-axis (columns) represent the stimulusV1 array and the y-axis (rows) represent the stimulusV2 array.
        """
        Pv1v2, xedges, yedges = np.histogram2d(x = stimulusV1[0, :], # index to get the 1D array out of the 2D array
                                               y = stimulusV2[0, :], # index to get the 1D array out of the 2D array
                                               bins = Nbins,
                                               range = [[min(stimulusV1edges), max(stimulusV1edges)], [min(stimulusV2edges), max(stimulusV2edges)]],
                                               density = True)
    
        # ----------- Unit Test Logic below for internal consistency----------------
        
        # Calculate bin widths in both dimensions
        x_bin_width = xedges[1:] - xedges[:-1]
        y_bin_width = yedges[1:] - yedges[:-1]

        # Calculate bin areas by taking the outer product of the bin widths
        bin_areas = np.outer(y_bin_width, x_bin_width)

        # Multiply each bin value with its corresponding bin area and sum over all bins - Definition per
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
        total = np.sum(Pv1v2 * bin_areas)
        
        # Assertion Test for internal consistency
        assert np.isclose(total, 1), 'The joint probability does not sum to 1'
        assert np.all(Pv1v2 >= 0), 'The joint probability density function has negative values'
        assert np.isclose(sum(np.sum(Pv1v2 * bin_areas , axis=0)), 1), 'The marginal probability of V1 does not sum to 1'
        assert np.isclose(sum(np.sum(Pv1v2 * bin_areas , axis=1)), 1), 'The marginal probability of V2 does not sum to 1'
        
        return Pv1v2, xedges, yedges

    @staticmethod
    def compute_marginal_prob(joint_prob):
        """
        Computes the marginal probability distribution of a joint probability distribution.
        
        Inputs:
        + Pv1v2: Joint probability table of size (Nbins, Nbins) the x-axis (columns) represent the stimulusV1 array and the y-axis (rows) represent the stimulusV2 array.
        """
        marginal_x = np.sum(joint_prob, axis=0)
        marginal_y = np.sum(joint_prob, axis=1)
        
        return marginal_x, marginal_y
    
    @staticmethod
    def compute_significance_between_pairs_of_tuning_curves_set(Nbins, observed_tf, expected_tf, observed_sem, expected_sem):
        """
        Computes the significance between pairs of tuning curves.
        
        NOTE: ARe the same bins used for both tuning curves? Check this
        """
        alpha = 0.05  # initial significance level
        num_tests = Nbins  # number of bins/tests
        alpha_adj = alpha / num_tests # Adjust alpha for Bonferroni correction
        z_score_adj = norm.ppf(1 - alpha_adj / 2) # Calculate z-score for adjusted alpha level using inverse of guassian CDF
        observed_confidence_interval = z_score_adj * observed_sem
        expected_confidence_interval = z_score_adj * expected_sem
        upper_bound_observed = observed_tf + observed_confidence_interval
        lower_bound_observed = observed_tf - observed_confidence_interval
        upper_bound_expected = expected_tf + expected_confidence_interval
        lower_bound_expected = expected_tf - expected_confidence_interval
        do_not_overlap = (upper_bound_observed < lower_bound_expected) | (lower_bound_observed > upper_bound_expected)
        return do_not_overlap
    
    @staticmethod
    def compute_binomial_chance_distribution(dictionary, Nbins):
        # count the number of successes (Trues) in each experiment
        counts = {k: np.sum(v) for k, v in dictionary.items()}

        # calculate proportions
        n_trials = Nbins  # number of trials in each experiment
        proportions = {k: v / n_trials for k, v in counts.items()}

        # estimate p as the mean of the proportions
        p_hat = np.mean(list(proportions.values()))

        # calculate a 95% confidence interval for the proportion
        z = norm.ppf(0.975)  # 1.96 for a 95% confidence interval
        conf_int = p_hat - z * np.sqrt((p_hat * (1 - p_hat)) / n_trials), p_hat + z * np.sqrt((p_hat * (1 - p_hat)) / n_trials)

        print(f'Estimated p: {p_hat}')
        print(f'95% confidence interval for p: {conf_int}')

        # plot the binomial distribution
        x = np.arange(n_trials + 1)
        pmf = binom.pmf(x, n_trials, p_hat)
        plt.stem(x, pmf, use_line_collection=True, basefmt=' ')
        plt.xlabel('Number of Trues')
        plt.ylabel('Probability')
        plt.title('Binomial Distribution')

        # plot the 95% confidence interval
        conf_int_scaled = np.array(conf_int) * n_trials  # scale to the number of trials
        plt.axvline(x=conf_int_scaled[1] + 0.5, color='red', linestyle='dashed')
        plt.show()

        # calculate the minimum number of successes needed to be in the upper 5% of the distribution
        min_successes_significant = binom.ppf(0.95, n_trials, p_hat)

        print(f'Minimum number of Trues for significance at the 5% level: {np.ceil(min_successes_significant)}')
        return min_successes_significant

def tunED_model_main(data, file_save_location):
    
    # sig_test_chanceofv2 = {}
    # Load chance significance data
    # with open('saved_dictionary_synthetic_v2chance.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    
    # min_successes_significant = TunEDModelStats.compute_binomial_chance_distribution(loaded_dict, Nbins=20)
    
    # sig_test_chanceofv1 = {}
    for cluster in range(37, 71): # NOTE - OFF by one error cluster zero doesn't exsist
        
        # setup
        filtered_df = data.filter(pl.col("spike_clusters") == cluster)
        Nsamples = len(filtered_df)
        # Ncells = 1
        Nbins = 20 # Number of bins to use to bin up the stimulus variable
        hdir = np.array(filtered_df["hdir"].to_numpy()).reshape(1, Nsamples)
        hsa  = np.array(filtered_df["hsa"].to_numpy()).reshape(1, Nsamples)
        raster = np.array(filtered_df["spike_count"].to_numpy()).reshape(1, Nsamples)
        
        # Log some information about the cluster
        print("The number of frames this cluster fired in:", Nsamples)
        print("This cluster has this number of spikes", sum(filtered_df["spike_count"]))
        print("The rho for the angles of this cluster is", np.corrcoef(hdir, hsa)[0, 1])
        
        # Calculate observed tuning curves --------------------------------------------------------
        hdir_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, 
                                                           stimulus_variable = hdir, 
                                                           Nbins = Nbins, 
                                                           Nsamples = Nsamples)
        
        hsa_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, 
                                                          stimulus_variable = hsa, 
                                                          Nbins = Nbins, 
                                                          Nsamples = Nsamples)
        
        # Calculate joint, marginal and conditional probabilities ---------------------------------
        jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(hdir, 
                                                                     hsa, 
                                                                     stimulusV2edges = hsa_tuning_object.stimulus_bin_edges, 
                                                                     stimulusV1edges = hdir_tuning_object.stimulus_bin_edges, 
                                                                     Nbins = Nbins)
        
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
        
        # Calculate expected NH tuning curves ------------------------------------------------------
        hdir_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hdir_tuning_object.tuning_func, 
                                                             observed_tuning_function_s2 = hdir_tuning_object.tuning_func_s2, 
                                                             num_values_for_Px = hsa_tuning_object.n, 
                                                             conditional_Py_x = Pv1_v2)
        
        hsa_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hsa_tuning_object.tuning_func, 
                                                             observed_tuning_function_s2 = hsa_tuning_object.tuning_func_s2, 
                                                             num_values_for_Px = hdir_tuning_object.n, 
                                                             conditional_Py_x = Pv2_v1)
        
        # Calculate the tuning curve significance -----------------------------------------------------------------------
        sig_testv2 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
                                                                                            observed_tf = hsa_tuning_object.tuning_func,
                                                                                                               expected_tf = hdir_NH_object.tuning_func_nh,
                                                                                                               observed_sem = hsa_tuning_object.tuning_func_sem, 
                                                                                                               expected_sem = hdir_NH_object.tuning_func_nh_sem)
        
        sig_testv1 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins,
                                                                                             observed_tf = hdir_tuning_object.tuning_func,
                                                                                             expected_tf = hsa_NH_object.tuning_func_nh,
                                                                                             observed_sem = hdir_tuning_object.tuning_func_sem,
                                                                                             expected_sem = hsa_NH_object.tuning_func_nh_sem)
        
        set_1_sig = False
        if np.sum(sig_testv1) > 13:
            set_1_sig = True
        
        set_2_sig = False
        if np.sum(sig_testv2) > 13:
            set_2_sig = True
        
        # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(23, 5))

        # Plot the first set of observed vs expected tuning curves
        ax[0].plot(hdir_tuning_object.bin_centres, 
                   hdir_tuning_object.tuning_func[0, :], 
                   '.-', 
                   label='Tuning to hdir', 
                   color="cornflowerblue")
        
        ax[0].fill_between(hdir_tuning_object.bin_centres, 
                           hdir_tuning_object.tuning_func[0, :] - hdir_tuning_object.tuning_func_sem[0, :], 
                           hdir_tuning_object.tuning_func[0, :] + hdir_tuning_object.tuning_func_sem[0, :], 
                           alpha=0.1, 
                           color="cornflowerblue")
        
        ax[0].plot(hdir_tuning_object.bin_centres, 
                   hsa_NH_object.tuning_func_nh[0, :], 
                   '.--', 
                   label='Tuning to hdir given NH that driver is hsa', 
                   color='darkorchid')
        
        ax[0].fill_between(hdir_tuning_object.bin_centres, 
                           hsa_NH_object.tuning_func_nh[0, :] - hsa_NH_object.tuning_func_nh_sem[0, :], 
                           hsa_NH_object.tuning_func_nh[0, :] + hsa_NH_object.tuning_func_nh_sem[0, :], 
                           alpha=0.1, 
                           color='darkorchid')
        
        ax[0].set_xlabel('v1')
        ax[0].set_ylabel('fr')
        ax[0].legend(loc='upper right')
        ax[0].set_title("Tuning to V1", fontweight="bold")
        
        # -----------------------------------------------------------------------------------------------

        # Plot the second set of observed vs expected tuning curves
        ax[1].plot(hsa_tuning_object.bin_centres, 
                   hsa_tuning_object.tuning_func[0, :], 
                   '.-', 
                   label='Tuning to hsa', 
                   color='cornflowerblue')
        
        ax[1].fill_between(hsa_tuning_object.bin_centres, 
                           hsa_tuning_object.tuning_func[0, :] -  hsa_tuning_object.tuning_func_sem[0, :], 
                           hsa_tuning_object.tuning_func[0, :] + hsa_tuning_object.tuning_func_sem[0, :], 
                           color='cornflowerblue', 
                           alpha=0.1)
        
        ax[1].plot(hsa_tuning_object.bin_centres, 
                   hdir_NH_object.tuning_func_nh[0, :], 
                   '.--', 
                   label='Tuning to hsa given NH that driver is hdir', 
                   color='darkorchid')
        
        ax[1].fill_between(hsa_tuning_object.bin_centres, 
                           hdir_NH_object.tuning_func_nh[0, :] - hdir_NH_object.tuning_func_nh_sem[0, :], 
                           hdir_NH_object.tuning_func_nh[0, :] + hdir_NH_object.tuning_func_nh_sem[0, :], 
                           color='darkorchid', alpha=0.1)
        
        ax[1].set_xlabel('v2')
        ax[1].set_ylabel('fr')
        ax[1].legend(loc='upper right')
        ax[1].set_title("Tuning to V2", fontweight="bold")
        
        # To verify that the data ingested is correct, generate a polar plot
        # angles must be in radians
        ax3 = fig.add_subplot(1, 3, 3, polar=True)
        bars = ax3.bar(hdir[0], raster[0]) # index to make into 1d array
        spikes = sum(filtered_df["spike_count"])
        plt.suptitle(f"Number of samples: {Nsamples}, V2 is the driving stimulus and V1 is the passenger stimulus. Cluster number {cluster}, spike number: {spikes}, corrcoeff: {np.corrcoef(hdir, hsa)[0, 1]}, is set 1 sig {set_1_sig}, is set 2 sig {set_2_sig}", fontweight="bold")
        
        plt.savefig(str(file_save_location) + f"cluster_{cluster}.png")
        # plt.show()


    # with open('saved_dictionary_synthetic_v1chance.pkl', 'wb') as f:
    #     pickle.dump(sig_test_chanceofv1, f)
    
if __name__ == '__main__':
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.
    """
    
    # Load chance significance data
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
        
    Ncells = 1
    Nsamples = 10000 # Number of samples to generate, i.e. number of frames
    Nbins = 10 # Number of bins to use to bin up the stimulus variable
    # min_successes_significant = TunEDModelStats.compute_binomial_chance_distribution(loaded_dict, Nbins = Nbins)
    
    # Generate stimuli
    stimulusV1 = np.random.randn(1, Nsamples) # Driver stimulus
    stimulusV2 = stimulusV1 * 0.2 + np.random.randn(1, Nsamples) # Passenger stimulus
    
    print(np.corrcoef(stimulusV1, stimulusV2)) # Print the stimulus variables as a correlation matrix, to show variable 2 is correlated with variable 1

    # Generate spike trains from a Poisson process with a rate that depends on the stimulus V1
    frate = 0.1*(stimulusV1[0, :] > 1.0) * stimulusV1[0, :]
    frate = np.minimum(1, frate)
    raster = np.zeros((1, Nsamples))
    raster[0, :] = np.random.poisson(frate)
    
    # Compute the tuning functions
    v1Object = ComputeObservedTuningFunction(raster, stimulusV1, Nbins, Nsamples)
    v2Object = ComputeObservedTuningFunction(raster, stimulusV2, Nbins, Nsamples)

    # Joint probability of the stimuli --------------------------------------------------------
    jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(stimulusV1, 
                                                                 stimulusV2, 
                                                                 stimulusV2edges = v2Object.stimulus_bin_edges, 
                                                                 stimulusV1edges = v1Object.stimulus_bin_edges,
                                                                 Nbins = Nbins)
    
    Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
    Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
    Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2) # Tranpose to ensure broadcasting works correctly row wise instead of column wise
    
    # ------------------------------- NULL Hypothesis tests ---------------------------------------------------------------------
    # apparent tuning to 'hdir' given NH that cell is driven purely by hsa:
    hdir_NH_object = ComputeNullHypothesisTuningFunction(v1Object.tuning_func, 
                                                         v1Object.tuning_func_s2, 
                                                         v2Object.n, 
                                                         Pv1_v2)
    
    # apparent tuning to 'hsa' given NH that cell is driven purely by hdir:
    hsa_NH_object = ComputeNullHypothesisTuningFunction(v2Object.tuning_func,
                                                        v2Object.tuning_func_s2,
                                                        v1Object.n,
                                                        Pv2_v1)
    
    # Compuete significance of tuning functions --------------------------------------------------------------------------------
    # Computes of the second set 
    # sig1 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
    #                                                                                observed_tf = v1Object.tuning_func,
    #                                                                                expected_tf = hsa_NH_object.tuning_func_nh,
    #                                                                                observed_sem = v1Object.tuning_func_sem, 
    #                                                                                expected_sem = hsa_NH_object.tuning_func_nh_sem,)
    
    # if np.sum(sig1) > 3:
    #     print("Significant difference in set 1!")
    # print(sig1)
        
    # sig2 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
    #                                                                                observed_tf = v2Object.tuning_func,
    #                                                                                expected_tf = hdir_NH_object.tuning_func_nh,
    #                                                                                observed_sem = v2Object.tuning_func_sem, 
    #                                                                                expected_sem = hdir_NH_object.tuning_func_nh_sem)
    
    # if np.sum(sig2) > 3:
    #     print("Significant difference in set 2!")
    # print(sig2)
    
    # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(23, 5))

    ax[0].plot(v1Object.bin_centres, 
            v1Object.tuning_func[0, :], 
            '.-', 
            label='Tuning to v1', 
            color="cornflowerblue")
    
    ax[0].fill_between(v1Object.bin_centres, 
                    v1Object.tuning_func[0, :] - v1Object.tuning_func_sem[0, :], 
                    v1Object.tuning_func[0, :] + v1Object.tuning_func_sem[0, :], 
                    alpha=0.1, 
                    color="cornflowerblue")
    
    ax[0].plot(v1Object.bin_centres, 
            hsa_NH_object.tuning_func_nh[0, :], 
            '.--', 
            label='Tuning to v1 given NH that driver is v2', 
            color='darkorchid')
    
    ax[0].fill_between(v1Object.bin_centres, 
                    hsa_NH_object.tuning_func_nh[0, :] - hsa_NH_object.tuning_func_nh_sem[0, :], 
                    hsa_NH_object.tuning_func_nh[0, :] + hsa_NH_object.tuning_func_nh_sem[0, :], 
                    alpha=0.1, 
                    color='darkorchid')
    
    ax[0].set_xlabel('v1')
    ax[0].set_ylabel('fr')
    ax[0].legend(loc='upper right')
    ax[0].set_title("Tuning to V1", fontweight="bold")
    
    # ------------------------------------------- second chart

    ax[1].plot(v2Object.bin_centres, 
            v2Object.tuning_func[0, :], 
            '.-', 
            label='Tuning to v2', 
            color='cornflowerblue')
    
    ax[1].fill_between(v2Object.bin_centres, 
                    v2Object.tuning_func[0, :] - v2Object.tuning_func_sem[0, :], 
                    v2Object.tuning_func[0, :] + v2Object.tuning_func_sem[0, :], 
                    color='cornflowerblue', 
                    alpha=0.1)
    
    ax[1].plot(v2Object.bin_centres, 
            hdir_NH_object.tuning_func_nh[0, :], 
            '.--', 
            label='Tuning to v2 given NH that driver is v1', 
            color='darkorchid')
    
    ax[1].fill_between(v2Object.bin_centres, 
                    hdir_NH_object.tuning_func_nh[0, :] - hdir_NH_object.tuning_func_nh_sem[0, :], 
                    hdir_NH_object.tuning_func_nh[0, :] + hdir_NH_object.tuning_func_nh_sem[0, :], 
                    color='darkorchid', 
                    alpha=0.1)
    
    ax[1].set_xlabel('v2')
    ax[1].set_ylabel('fr')
    ax[1].legend(loc='upper right')
    ax[1].set_title("Tuning to V2", fontweight="bold")
    
    plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
    plt.show()
    
    # with open('saved_dictionary.pkl', 'wb') as f:
    #     pickle.dump(significane, f)