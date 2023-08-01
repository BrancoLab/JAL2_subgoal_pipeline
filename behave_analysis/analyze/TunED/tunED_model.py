"""
####### Overview of the TunED model #######

Tuning function (tf): 
+ Mean firing rate of a neuron to a given stimulus µ(v1) or µ(v2). See class compute_observed_tuning_function for more details.

Null hypothesis (nh): 
+ We set a null hypothesis asking how much of the tuning function of one neuron to stimulus V1 can be explained by a second stimulus variable V2 and its interaction with V1. We compute the expected
  conditional E[fr(v2)|v1]. And thus the NH is that the tuning function of that neuron is purely driven by V2 and not V1. See class compute_null_hypothesis for more details. If the NH
  is true then the tuning function of that neuron should be the same as the expected conditional.

TODO:
+ There should be some quality checks done on the ingested data because I found a spike count at 130  in one frame for one cell which is impossible so data quality is not there yet.
+ Bins are not uniformly sampling with some bins empty - this might be causing unknown issues
"""

# OS Imports
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from scipy.stats import norm, binom
from behave_analysis.analyze.linshit import LinearShift
import pickle
import os
from tqdm import tqdm

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
                    n[cluster_idx, bin_idx] = np.sum(mask) # how many samples are in this bin
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
    def compute_conditional_probabilities(joint, marginal_x, marginal_y):
        """
        Compute the conditional probabilities of the joint probability distribution. As a reminder, given the different shapes of the arrays numpy has to broadcast the arrays to the shape of the matrix.
        Broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html - Force reshape is required to prevent unwanted broadcasting.
        
        So to get the conditional probability P(X|Y) for each cell, you divide the cell's value (P(X, Y)) by the sum of its column (P(Y)). 
        This is equivalent to dividing the entire 2D histogram (all cells) by the 1D array that represents the sums of columns.
        
        # NOTE: Not used, but keeping it here for future reference in case want to figure it out
        """
        marginal_x_reshaped = marginal_x[np.newaxis, :] # Turn into a row vector of shape (Nbins, 1)
        marginal_y_reshaped = marginal_y[np.newaxis, :] #  Turn into a row vector of shape (Nbins, 1)
        
        conditional_x_given_y = joint.T / marginal_y_reshaped 
        conditional_y_given_x = joint / marginal_x_reshaped
        
        raise NotImplementedError('Was trying to implement the conditional probabilities but I am not sure if this is correct.')
    
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
    def compute_binomial_chance_distribution(dictionary, Nbins = 20):
              
        # Sum up the number of significant bins for each cluster
        significantBins = {cluster: np.sum(v) for cluster, v in dictionary.items()}
        
        # If there are no sinificant bins assume that cluster is noise and exlude it from the analysis
        significantBins = {key: value for key, value in significantBins.items() if value > 0}

        # For each of those Trues, divided by the total number of bins to get the proportion of significant bins
        proportions = [count / (len(dictionary[key]) * 20) for key, count in significantBins.items()]

        # Estimate p as the mean of the proportions, which is the probability of bin being significant
        p_hat = np.mean(proportions)
        assert 0 <= p_hat <= 1, 'Estimated p is not between 0 and 1'
        logger.info(f'Estimated p: {p_hat} (probability of a bin being significant by chance')

        # Calculate a 95% confidence interval for the proportion
        z = norm.ppf(0.975)  # for a 95% confidence interval
        conf_int = p_hat - z * np.sqrt((p_hat * (1 - p_hat)) / Nbins), p_hat + z * np.sqrt((p_hat * (1 - p_hat)) / Nbins)

        logger.info(f'95% confidence interval for p: {conf_int} (ranged probability of a single bin for chance')

        # Plot the binomial distribution
        plt.figure(figsize=(10, 5))
        x = np.arange(Nbins + 1)
        pmf = binom.pmf(x, Nbins, p_hat)
        plt.stem(x, pmf, use_line_collection=True, basefmt=' ')
        plt.xlabel('Number of successes')
        plt.ylabel('Probability')
        plt.title('Binomial Distribution')

        # Plot the 95% confidence interval
        conf_int_scaled = np.array(conf_int) * Nbins
        plt.axvline(x=conf_int_scaled[1] + 0.5, color='red', linestyle='dashed')
        plt.show()

        # Calculate the minimum number of successes needed to be in the upper 5% of the distribution
        min_successes_significant = binom.ppf(0.95, Nbins, p_hat)
        logger.info(f'Minimum number of successes for significance at the 5% level: {np.ceil(min_successes_significant)}')

        return min_successes_significant

    @staticmethod 
    def compute_synthetic_accuracy(dictionary, number_of_cells_produced_per_angle):
        """
        Computes the accuracy of the model by computing the percentage of cells that are correctly predicted.
        """
        # Extract the different categories of tunned cells 
        first_set_of_cells = {k: v for k, v in dictionary.items() if k < number_of_cells_produced_per_angle}
        second_set_of_cells = {k: v for k, v in dictionary.items() if k >= number_of_cells_produced_per_angle if k < number_of_cells_produced_per_angle * 2} # hack
        
        # Compute total num samples
        # first_set_total_samples = np.sum(list(first_set_of_cells.values()))
        # second_set_total_samples = np.sum(list(second_set_of_cells.values()))
        
        first_set_total_samples = 37
        second_set_total_samples = 72 - 37
        
        # First set -----------------------------------------------
        
        # Compute accuracy by summing up all the True values
        setOneCorrectCount = 0
        for key, value in first_set_of_cells.items():
          if value == [True, False]:
            setOneCorrectCount += 1
        setOneAccuracy = setOneCorrectCount / first_set_total_samples
        
        setOneInccorectCount = 0
        for key, value in first_set_of_cells.items():
          if value == [False, True]:
            setOneInccorectCount += 1
        setOneInaccuracy = setOneInccorectCount / first_set_total_samples
            
        # Print accuracies
        print(f"Accuracy of the first set of cells: {setOneAccuracy}")
        print(f"Inaccuracy of the first set of cells: {setOneInaccuracy}")
        
        # Second set -----------------------------------------------
        setTwoCorrectCount = 0
        for key, value in second_set_of_cells.items():
          if value == [False, True]:
            setTwoCorrectCount += 1
        setTwoAccuracy = setTwoCorrectCount / second_set_total_samples
        
        setTwoInccorectCount = 0
        for key, value in second_set_of_cells.items():
          if value == [True, False]:
            setTwoInccorectCount += 1
        setTwoInaccuracy = setTwoInccorectCount / second_set_total_samples
        
        # Print accuracies 
        print(f"Accuracy of the second set of cells: {setTwoAccuracy}")
        print(f"Inaccuracy of the second set of cells: {setTwoInaccuracy}")
        
class TunEdModel:
    def __init__(self, 
                 inherited_object, 
                 analyze_efizz_settings, 
                 save_location, 
                 save_plots = False, 
                 apply_linear_shift = False):
        
        self.settings = analyze_efizz_settings
        self.inherited_object = inherited_object
        self.directory_location = save_location
        self.apply_linear_shift = apply_linear_shift
        self.data_df = self.filter_data_by_period() # before shelter or after shelter etc
        
        if not self.apply_linear_shift: 
            assert os.path.exists('linear_shift_null_distribution_binomial.pkl'), "File does not exist! You must run with lin shift first to generate the null"
            logger.info("Loading the null distribution for a previously computed binomial test")
            self.accuracy_dic = self.execute_model_per_cluster(save_plots)
        
        if self.apply_linear_shift:
            self.accuracy_dic = self.excute_model_per_cluster_with_linear_shift(shifted_variale = "hsa")
        
        # TunEDModelStats.compute_synthetic_accuracy(self.accuracy_dic, number_of_cells_produced_per_angle = 37)
        
    def filter_data_by_period(self):
        """
        The purpose of this function is to filter the data by the period of interest but also to remove the data that is not relevant to the model such as 
        escapse periods and periods when the mouse is in the shelter.
        """
        
        if self.settings.analyze_only_the_period_before_shelter & self.settings.analyze_only_the_period_before_barrier:
            assert False, "Cannot analyze only the period before the shelter and the period before the barrier at the same time."
        
        # Filter out escape, periods when the mouse is in his house and periods when the shelter is not present
        if self.settings.analyze_only_the_period_before_shelter:
            filtered_data = self.inherited_object.data_df.filter((self.inherited_object.data_df["OutofshelterIdx"] == True) & 
                                                                 (self.inherited_object.data_df["EscapePeriod"] == False) &
                                                                 (self.inherited_object.data_df["shelter_only"] == False))
        
        # Filter out escape, and periods when the mouse is in his house, and periods when the shelter is and is not present
        if not self.settings.analyze_only_the_period_before_shelter:
            filtered_data = self.inherited_object.data_df.filter((self.inherited_object.data_df["OutofshelterIdx"] == True) & 
                                                                 (self.inherited_object.data_df["EscapePeriod"] == False))
            logger.info("Analysing the whole session with escapes and periods when the mouse is in his house removed")
        
        # Filter on the period just before the barrier
        if self.settings.analyze_only_the_period_before_barrier:
            filtered_data = self.inherited_object.data_df.filter((self.inherited_object.data_df["barrier_present"] == False))
        
        
        return filtered_data
    
    def init_model_inputs(self, data_df):
        Nsamples = len(data_df)
        Nbins = 20 # Number of bins to use to bin up the stimulus variable
        hdir = np.array(data_df["hdir"].to_numpy()).reshape(1, Nsamples)
        hsa  = np.array(data_df["hsa"].to_numpy()).reshape(1, Nsamples)
        raster = np.array(data_df["spike_count"].to_numpy()).reshape(1, Nsamples)
        return Nsamples, Nbins, hdir, hsa, raster
    
    def produce_bool_of_signifiance(self, hdir_sig, hsa_sig, num_bins_required_to_be_significant = 18):
        """
        The purpose of this function is to produce a boolean that indicates whether the tuning functions are significantly different or not.
        """
        is_hdir_sig = False
        if np.sum(hdir_sig) > num_bins_required_to_be_significant:
            is_hdir_sig = True
            logger.success("The tuning function for head direction is significantly different to the null hypothesis")
        
        is_hsa_sig = False
        if np.sum(hsa_sig) > num_bins_required_to_be_significant:
            is_hsa_sig = True
            logger.success("The tuning function for head shelter angle is significantly different to the null hypothesis")
        
        return is_hdir_sig, is_hsa_sig
    
    def plot_and_save_tuning_functions(self, hdir_tuning_object, hsa_NH_object, hsa_tuning_object, hdir_NH_object, filtered_df, cluster, hdir, hsa, is_hdir_sig, is_hsa_sig, Nsamples):
        fig, ax = plt.subplots(1, 2, figsize=(23, 5))
        
        # Plot hdir observed vs null tuning function
        ax[0].set_title("Tuning to head direction and the NH that the driver is head shelter angle", fontweight="bold")
        ax[0].plot(hdir_tuning_object.bin_centres, hdir_tuning_object.tuning_func[0, :], '.-', label='Tuning to hdir', color="cornflowerblue")
        ax[0].fill_between(hdir_tuning_object.bin_centres, hdir_tuning_object.tuning_func[0, :] - hdir_tuning_object.tuning_func_sem[0, :], 
                           hdir_tuning_object.tuning_func[0, :] + hdir_tuning_object.tuning_func_sem[0, :], alpha=0.1, color="cornflowerblue")
        ax[0].plot(hdir_tuning_object.bin_centres, hsa_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to hdir given NH that driver is hsa', color='darkorchid')
        ax[0].fill_between(hdir_tuning_object.bin_centres, hsa_NH_object.tuning_func_nh[0, :] - hsa_NH_object.tuning_func_nh_sem[0, :], 
                           hsa_NH_object.tuning_func_nh[0, :] + hsa_NH_object.tuning_func_nh_sem[0, :], alpha=0.1, color='darkorchid')
        ax[0].set_xlabel('Radians')
        ax[0].set_ylabel('fr')
        ax[0].legend(loc='upper right')
        
        # Plot hsa observed vs null tuning function
        ax[1].set_title("Tuning to shelter angle and the NH that the driver is head direction", fontweight="bold")
        ax[1].plot(hsa_tuning_object.bin_centres, hsa_tuning_object.tuning_func[0, :], '.-', label='Tuning to hsa', color='cornflowerblue')
        ax[1].fill_between(hsa_tuning_object.bin_centres, hsa_tuning_object.tuning_func[0, :] -  hsa_tuning_object.tuning_func_sem[0, :], 
                           hsa_tuning_object.tuning_func[0, :] + hsa_tuning_object.tuning_func_sem[0, :], color='cornflowerblue', alpha=0.1)
        ax[1].plot(hsa_tuning_object.bin_centres, hdir_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to hsa given NH that driver is hdir', color='darkorchid')
        ax[1].fill_between(hsa_tuning_object.bin_centres, hdir_NH_object.tuning_func_nh[0, :] - hdir_NH_object.tuning_func_nh_sem[0, :], 
                           hdir_NH_object.tuning_func_nh[0, :] + hdir_NH_object.tuning_func_nh_sem[0, :], color='darkorchid', alpha=0.1)
        ax[1].set_xlabel('Radians')
        ax[1].set_ylabel('fr')
        ax[1].legend(loc='upper right')
        
        # Titles and saving the figure
        spikes = sum(filtered_df["spike_count"])
        plt.suptitle(f" Number of samples: {Nsamples}, V2 is the driving stimulus and V1 is the passenger stimulus. \
                       Cluster number {cluster}, spike number: {spikes}, corrcoeff: {np.corrcoef(hdir, hsa)[0, 1]}, is set 1 sig {is_hdir_sig}, is set 2 sig {is_hsa_sig}", 
                       fontweight="bold")
        plt.show()
        # plt.savefig(str(self.directory_location) + "\\" + f"_cluster_{cluster}.png")
    
    def execute_model_per_cluster(self, save_plots = True):
        """
        The purpose of this function is to execute the TunEd model for each cluster in the data and thus calls all of the relevant classes and functions to do so.
        """
        
        with open("linear_shift_null_distribution_binomial.pkl", 'rb') as f:
            linear_shift_null_distribution_bionomial = pickle.load(f)
        
        min_bins_for_sig = TunEDModelStats.compute_binomial_chance_distribution(linear_shift_null_distribution_bionomial)
        # min_bins_for_sig = 16 # Overriding the min bins for sig to be 16
        
        # Init params for model
        accuracy_dic = {} # Dict to store the accuracy of the model for each cluster
        bool_array_set_1 = {} # An array of bools informing significance of observed to null tuning function for hdir
        bool_array_set_2 = {} # An array of bools informing significance of observed to null tuning function for hsa
        
        for cluster in np.unique(self.inherited_object.data_df["spike_clusters"]):
            
   
            filtered_df = self.data_df.filter(pl.col("spike_clusters") == cluster)
            Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)
            logger.info(f"Running TunEd model for cluster {cluster} active for {Nsamples} firing a total of {sum(filtered_df['spike_count'])} spikes, with a correlation coefficient of {np.corrcoef(hdir, hsa)[0, 1]}")
            
            if sum(filtered_df['spike_count']) == 0:
                logger.warning("No spikes in this cluster, skipping, should not be the case")
                continue
            
            # Compute observed tuning functions
            hdir_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hdir, Nbins = Nbins, Nsamples = Nsamples)
            hsa_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hsa, Nbins = Nbins, Nsamples = Nsamples)
            
            # Compute probabilities
            jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(hdir, hsa, stimulusV2edges = hsa_tuning_object.stimulus_bin_edges, stimulusV1edges = hdir_tuning_object.stimulus_bin_edges, Nbins = Nbins)
            Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
            Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
            Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
            
            # Compute NULL hypothesis that the driver is purely V1
            hdir_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hdir_tuning_object.tuning_func, 
                                                                 observed_tuning_function_s2 = hdir_tuning_object.tuning_func_s2, 
                                                                 num_values_for_Px = hsa_tuning_object.n, 
                                                                 conditional_Py_x = Pv1_v2)
            
            # Compute the NULL hypothesis that the driver is purely V2
            hsa_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hsa_tuning_object.tuning_func, 
                                                                observed_tuning_function_s2 = hsa_tuning_object.tuning_func_s2, 
                                                                num_values_for_Px = hdir_tuning_object.n, 
                                                                conditional_Py_x = Pv2_v1)
            
            # Compute the significance of difference between the observed and expected tuning functions
            hdir_significance = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins,
                                                                                                        observed_tf = hdir_tuning_object.tuning_func,
                                                                                                        expected_tf = hsa_NH_object.tuning_func_nh,
                                                                                                        observed_sem = hdir_tuning_object.tuning_func_sem,
                                                                                                        expected_sem = hsa_NH_object.tuning_func_nh_sem)
            
            hsa_significance = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
                                                                                                       observed_tf = hsa_tuning_object.tuning_func,
                                                                                                       expected_tf = hdir_NH_object.tuning_func_nh,
                                                                                                       observed_sem = hsa_tuning_object.tuning_func_sem, 
                                                                                                       expected_sem = hdir_NH_object.tuning_func_nh_sem)
            
            # Produce boolean of significance
            is_hdir_sig, is_hsa_sig = self.produce_bool_of_signifiance(hdir_significance, 
                                                                       hsa_significance,
                                                                       num_bins_required_to_be_significant = min_bins_for_sig)
            accuracy_dic[cluster] = [is_hdir_sig, is_hsa_sig]
            
            # Plot and save tuning functions
            if save_plots:
                self.plot_and_save_tuning_functions(hdir_tuning_object, 
                                                    hsa_NH_object, 
                                                    hsa_tuning_object, 
                                                    hdir_NH_object, 
                                                    filtered_df, 
                                                    cluster, 
                                                    hdir, 
                                                    hsa, 
                                                    is_hdir_sig, 
                                                    is_hsa_sig,
                                                    Nsamples)
            
       
        
        
        # ------------------------Compute the number of clusters classified as hdir and hsa --------------------------------------
        hdir_classified = 0
        for key, value in accuracy_dic.items(): 
            if value == [True, False]: 
                hdir_classified += 1
        
        hsa_classified = 0
        for key, value in accuracy_dic.items():
          if value == [False, True]:
            hsa_classified += 1
            
        logger.info(f"Number of clusters classified as hdir: {hdir_classified}")
        logger.info(f"Number of clusters classified as hsa: {hsa_classified}")
        
        return None
    
    # Linear shift extensions
    
    def tuned_model_user_defined_function_for_linear_shift(self, X, y):
        
        # Prepare data
        filtered_df = X
        filtered_df = filtered_df.with_column(y) # Replace the NH column with the shifted NH column
        Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)
        
        # Compute observed tuning functions
        hdir_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hdir, Nbins = Nbins, Nsamples = Nsamples)
        hsa_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hsa, Nbins = Nbins, Nsamples = Nsamples)
        
        # Compute probabilities
        jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(hdir, hsa, stimulusV2edges = hsa_tuning_object.stimulus_bin_edges, stimulusV1edges = hdir_tuning_object.stimulus_bin_edges, Nbins = Nbins)
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
        
        # Compute NULL hypothesis that the driver is purely V1
        hdir_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hdir_tuning_object.tuning_func, 
                                                             observed_tuning_function_s2 = hdir_tuning_object.tuning_func_s2, 
                                                             num_values_for_Px = hsa_tuning_object.n, 
                                                             conditional_Py_x = Pv1_v2)
        
        # Compute the NULL hypothesis that the driver is purely V2
        hsa_NH_object = ComputeNullHypothesisTuningFunction(observed_tuning_function = hsa_tuning_object.tuning_func, 
                                                            observed_tuning_function_s2 = hsa_tuning_object.tuning_func_s2, 
                                                            num_values_for_Px = hdir_tuning_object.n, 
                                                            conditional_Py_x = Pv2_v1)
        
        # Compute the significance of difference between the observed and expected tuning functions
        hdir_significance = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins,
                                                                                                    observed_tf = hdir_tuning_object.tuning_func,
                                                                                                    expected_tf = hsa_NH_object.tuning_func_nh,
                                                                                                    observed_sem = hdir_tuning_object.tuning_func_sem,
                                                                                                    expected_sem = hsa_NH_object.tuning_func_nh_sem)
        
        hsa_significance =  TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
                                                                                                    observed_tf = hsa_tuning_object.tuning_func,
                                                                                                    expected_tf = hdir_NH_object.tuning_func_nh,
                                                                                                    observed_sem = hsa_tuning_object.tuning_func_sem, 
                                                                                                    expected_sem = hdir_NH_object.tuning_func_nh_sem)
        if 0: # Plotting for debugging
            cluster, is_hdir_sig, is_hsa_sig = None, None, None
            self.plot_and_save_tuning_functions(hdir_tuning_object, 
                                                hsa_NH_object, 
                                                hsa_tuning_object, 
                                                hdir_NH_object, 
                                                filtered_df, 
                                                cluster, 
                                                hdir, 
                                                hsa, 
                                                is_hdir_sig, 
                                                is_hsa_sig,
                                                Nsamples)
        
        # have I got it the right way around?
        if y.name == "hdir":
            return np.sum(hsa_significance[0])
        
        elif y.name == "hsa":
            return np.sum(hdir_significance[0])
        
    def excute_model_per_cluster_with_linear_shift(self, shifted_variale = "hsa"):
        full_bionmial = {}
        clusters = np.unique(self.inherited_object.data_df["spike_clusters"])
        for cluster in tqdm(clusters, desc="Genereating null distribution for linear shift per cluster"):
            X = self.data_df.filter(pl.col("spike_clusters") == cluster)
            result = LinearShift(X =  X, 
                                 y = X[shifted_variale], 
                                 stat_computation_func = self.tuned_model_user_defined_function_for_linear_shift,
                                 size_of_central_chunk = int(len(X) / 3))
            full_bionmial[cluster] = result.pseudo_stats
        
        # Save the results
        with open('linear_shift_null_distribution_binomial.pkl', 'wb') as f:
            pickle.dump(full_bionmial, f)
                        
        return full_bionmial
           
if __name__ == '__main__':
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.
    """
    
    # Load chance significance data
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
        
    Ncells = 1
    Nsamples = 100000 # Number of samples to generate, i.e. number of frames
    Nbins = 10 # Number of bins to use to bin up the stimulus variable
    # min_successes_significant = TunEDModelStats.compute_binomial_chance_distribution(loaded_dict, Nbins = Nbins)
    
    # Generate stimuli
    stimulusV1 = np.random.randn(1, Nsamples) # Driver stimulus
    stimulusV2 = stimulusV1 * 0.6 + np.random.randn(1, Nsamples) # Passenger stimulus
    
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
    
    Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1) # TODO write a compute conditional probability function with logic i Understand
    Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2) # Tranpose to ensure broadcasting works correctly row wise instead of column wise
    
    # ------------------------------- NULL Hypothesis tests ---------------------------------------------------------------------
    V2_NH_object = ComputeNullHypothesisTuningFunction(v2Object.tuning_func, v2Object.tuning_func_s2, v1Object.n, Pv2_v1) # E[fr(v2)|v1] given NH that cell is driven purely by V2:
    V1_NH_object = ComputeNullHypothesisTuningFunction(v1Object.tuning_func, v1Object.tuning_func_s2, v2Object.n, Pv1_v2) # E[fr(v1)|v2] given NH that cell is driven purely by V1:
    # ---------------------------------------------------------------------------------------------------------------------------
    
    # Compuete significance of tuning functions --------------------------------------------------------------------------------
    # Computes of the second set 
    # sig1 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
    #                                                                                observed_tf = v1Object.tuning_func,
    #                                                                                expected_tf = V1_NH_object.tuning_func_nh,
    #                                                                                observed_sem = v1Object.tuning_func_sem, 
    #                                                                                expected_sem = V1_NH_object.tuning_func_nh_sem,)
    
    # if np.sum(sig1) > 3:
    #     print("Significant difference in set 1!")
    # print(sig1)
        
    # sig2 = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
    #                                                                                observed_tf = v2Object.tuning_func,
    #                                                                                expected_tf = V2_NH_object.tuning_func_nh,
    #                                                                                observed_sem = v2Object.tuning_func_sem, 
    #                                                                                expected_sem = V2_NH_object.tuning_func_nh_sem)
    
    # if np.sum(sig2) > 3:
    #     print("Significant difference in set 2!")
    # print(sig2)
    
    # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))

    ax[0].plot(v1Object.bin_centres, v1Object.tuning_func[0, :], '.-', label='Tuning to v1', color="cornflowerblue")
    ax[0].fill_between(v1Object.bin_centres, v1Object.tuning_func[0, :] - v1Object.tuning_func_sem[0, :], v1Object.tuning_func[0, :] + v1Object.tuning_func_sem[0, :], alpha=0.1, color="cornflowerblue")
    ax[0].plot(v1Object.bin_centres, V2_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to v1 given NH that driver is v2 E[fr(v2)|v1]', color='darkorchid')
    ax[0].fill_between(v1Object.bin_centres, V2_NH_object.tuning_func_nh[0, :] - V2_NH_object.tuning_func_nh_sem[0, :], V2_NH_object.tuning_func_nh[0, :] + V2_NH_object.tuning_func_nh_sem[0, :], alpha=0.1, color='darkorchid')
    ax[0].set_xlabel('v1')
    ax[0].set_ylabel('fr')
    ax[0].legend(loc='upper right')
    ax[0].set_title("Tuning to V1", fontweight="bold")
    
    # ------------------------------------------- second chart ------------------------------------------------------

    ax[1].plot(v2Object.bin_centres, v2Object.tuning_func[0, :], '.-', label='Tuning to v2', color='cornflowerblue') 
    ax[1].fill_between(v2Object.bin_centres, v2Object.tuning_func[0, :] - v2Object.tuning_func_sem[0, :], v2Object.tuning_func[0, :] + v2Object.tuning_func_sem[0, :], color='cornflowerblue', alpha=0.1)
    ax[1].plot(v2Object.bin_centres, V1_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to v2 given NH that driver is v1 E[fr(v1)|v2]', color='darkorchid')
    ax[1].fill_between(v2Object.bin_centres, V1_NH_object.tuning_func_nh[0, :] - V1_NH_object.tuning_func_nh_sem[0, :], V1_NH_object.tuning_func_nh[0, :] + V1_NH_object.tuning_func_nh_sem[0, :], color='darkorchid', alpha=0.1)
    ax[1].set_xlabel('v2')
    ax[1].set_ylabel('fr')
    ax[1].legend(loc='upper right')
    ax[1].set_title("Tuning to V2", fontweight="bold")
    
    plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
    plt.show()
    
    # with open('saved_dictionary.pkl', 'wb') as f:
    #     pickle.dump(significane, f)