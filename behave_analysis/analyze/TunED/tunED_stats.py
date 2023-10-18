# OS Libaries

import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
from loguru import logger


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
        
        Significance computation explained:
        (1) Alpha calulcation: First correct the alpha using the Bonferroni correction
        (2) z-score calculation: Feed this alpha into the inverse of the guassian CDF to get the z-score, given we care about the central 95% of the distribution
        and the CDF includes the left side of the distribution, we divide alpha by 2 to get the central 95% of the distribution. 0.05 / 2 = 0.025
        as we need 2.5% on each side of the distribution. We do 1- alpha to get the right side of the distribution. 1 - 0.025 = 0.975. It's
        called z-score because it's the number of standard deviations away from the mean.
        
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
        return do_not_overlap, observed_confidence_interval, expected_confidence_interval
    
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