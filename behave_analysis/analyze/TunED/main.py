# OS Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from loguru import logger
from scipy.spatial import distance
from scipy.stats import norm


# TODO - There should be some quality checks done on the ingested data because I found a spike count at 130 
# in one frame for one cell which is impossible so data quality is not there yet.
# Putting here so don't forget. It could be somelike checking that there are no spike counts about 10 or something basic

class TunED:
    """
    Estimate tuning functions for 2 stimulus variables, along with tuning functions under the null hypothesis (NH) that only one variable is the driver.
    A new instance of this class should be created for each stimulus variable.
    """
    def __init__(self, spike_count_matrix, stimulus_variable, Nbins, Nsamples):
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
        self.Nsamples = Nsamples
        
        self.__conduct_input_validation()
        self.stimulus_idx, self.stimulus_bin_edges = self.compute_stimulus_indx()
        
    def __conduct_input_validation(self) -> None:
        """
        A private method to validate the input data. Raises a ValueError if the input data is not as expected.
        """
        if self.Nsamples != self.stimulus_variable.shape[1]:
            raise ValueError('Mismatched input')
        if self.stimulus_variable.shape[0] != 1:
            raise ValueError('Unexpected input format')
    
    def compute_stimulus_indx(self):
        """
        First: Using a requested number of bins. Compute the bin edges later used for binning and plotting.
        Second: Using those bin edges, assign a bin index to each event recorded during a frame

        Returns:
            stimulus_idx: <np.ndarray> of size (1, Nsamples). The stimulus index for each sample.
            stimulus_bin_edges: <np.ndarray> of size (Nbins, ). The bin edges used to bin the stimulus variable.
        """
        stimulus_bin_edges =  np.linspace(np.min(self.stimulus_variable), np.max(self.stimulus_variable), self.Nbins+1)
        stimulus_idx = np.digitize(self.stimulus_variable, stimulus_bin_edges) - 1
        
        # print(f"Stimulus bin idx is {stimulus_idx}")
        # print(f"The bin edges are {stimulus_bin_edges}")
        
        #  # Plotting the histogram
        # fig = plt.figure()
        # plt.hist(stimulus_idx[0], bins=self.Nbins, edgecolor='black')
        # plt.title("Histogram of stimulus index")
        # plt.xlabel("Stimulus index")
        # plt.ylabel("Frequency")
        # plt.show()
        
        # assert max(stimulus_bin_edges) < 3.15, "Largest bin impossible for radians"
        # assert min(stimulus_bin_edges) > -3.15, "Smallest bin impossible for radians"
        
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
            tf_s2: <np.ndarray> of size (Ncells, Nbins). The variance of the tuning function for each cell. - Second moment
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
                    logger.error(f"Cluster {cluster_idx} has no samples in bin {bin_idx}")
                    tf[cluster_idx, bin_idx] = -1
                    tf_s2[cluster_idx, bin_idx] = 0
                    tf_sem[cluster_idx, bin_idx] = 0
                    n[cluster_idx, bin_idx] = 0
                    
        bin_centres = (self.stimulus_bin_edges[1:] + self.stimulus_bin_edges[:-1]) / 2
        
        return tf, tf_sem, tf_s2, n, bin_centres
    
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
            Pv2v1, _, _: Joint probability table of size (Nbins, Nbins)
        """
        Pv2v1, xedges, yedges = np.histogram2d(x = stimulusV1[0, :], # index to get the 1D array out of the 2D array
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
        total = np.sum(Pv2v1 * bin_areas)
        
        # Assertion Test for internal consistency
        assert np.isclose(total, 1), 'The joint probability does not sum to 1'
        assert np.all(Pv2v1 >= 0), 'The joint probability density function has negative values'
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

    @staticmethod
    def compute_significance_between_pairs_of_tuning_curves(Nbins, observed_tf, expected_tf, observed_sem, expected_sem):
        """
        Computes the significance between pairs of tuning curves.
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
    
# Does everything the below does but now can import it to test module
def tunED_main(dataframe, save_plot_location):

    for cluster in range(1, 70): # NOTE - OFF by one error cluster zero doesn't exsist
        
        # Filter by one cluster to test
        filtered_df = dataframe.filter(pl.col("spike_clusters") == cluster)
        
        # print how many spikes that cluster has
        print("This cluster has this number of spikes", sum(filtered_df["spike_count"]))
        
        # Parameters
        Nsamples = len(filtered_df)
        Ncells = 1
        Nbins = 20 # Number of bins to use to bin up the stimulus variable
        print("The number of frames this cluster fired in:", Nsamples)
        
        # Extract the stimulus variables
        hdir = np.array(filtered_df["hdir"].to_numpy()).reshape(1, Nsamples)
        hsa  = np.array(filtered_df["hsa"].to_numpy()).reshape(1, Nsamples)
        
        print("The rho for the angles of this cluster is", np.corrcoef(hdir, hsa)[0, 1])
        
        raster = np.array(filtered_df["spike_count"].to_numpy()).reshape(1, Nsamples)
        
        v1Object = TunED(raster, hdir, Nbins, Nsamples)
        tuning_function_hdir, tf_semv1, tf_s2v1, nv1, bin_centresV1 = v1Object.tuning_function(Ncells, Nbins, raster)
        v2Object = TunED(raster, hsa, Nbins, Nsamples)
        tuning_function_hsa, tf_semv2, tf_s2v2, nv2, bin_centresV2 = v2Object.tuning_function(Ncells, Nbins, raster)

        # Joint probability of the stimuli --------------------------------------------------------
        jointProb_stimuli, _, _ = TunED.compute_joint_prob(hdir, 
                                                           hsa, 
                                                           stimulusV2edges = v2Object.stimulus_bin_edges, 
                                                           stimulusV1edges = v1Object.stimulus_bin_edges,
                                                           Nbins = Nbins)
            
        # Marginalize the joint probability ------------------------------------------------------
        Pv1, Pv2 = TunED.marginalize(jointProb_stimuli)
        
        # Compute the conditional probabilities for the stimulus ---------------------------------------------------
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
        # Tranpose to ensure broadcasting works correctly row wise instead of column wise
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
        
        # NULL Hypothesis tests ---------------------------------------------------------------------
        
        # apparent tuning to 'v2' given NH that cell is driven purely by v1 P(v2|v1):
        tf_x_nh2_driven_by_1, tf_x_nh_sem21 = TunED.tuning_function_null_hypothesis_PYX(tuning_function_hdir, tf_s2v1, nv2, Pv1_v2)
        
        # apparent tuning to 'v1' given NH that cell is driven purely by v2 P(v1|v2):
        tf_x_nh1_drive_by_2, tf_x_nh_sem12 = TunED.tuning_function_null_hypothesis_PYX(tuning_function_hsa, tf_s2v2, nv1, Pv2_v1)
        
        # Test significance between pairs of tuning curves ------------------------------------------
        significance_v1 = TunED.compute_significance_between_pairs_of_tuning_curves(Nbins, tuning_function_hdir, tf_x_nh1_drive_by_2, tf_semv1, tf_x_nh_sem12)
        significance_v2 = TunED.compute_significance_between_pairs_of_tuning_curves(Nbins, tuning_function_hsa, tf_x_nh2_driven_by_1, tf_semv2, tf_x_nh_sem21)
        
        # print("The significance of the tuning curve for hdir given the null hypothesis that the cell is driven purely by hsa is (True if different)", significance_v1)
        # print("The significance of the tuning curve for hsa given the null hypothesis that the cell is driven purely by hdir is (True if different)", significance_v2)
        
        hdir_count_sig = sum(sum(significance_v1))
        hsa_count_sig = sum(sum(significance_v2))
        
        if hdir_count_sig > hsa_count_sig:
            print("This cell is more significant to being driven by hdir")
        
        else:
            print("This cell is more significant to being driven by hsa")
        
        # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(23, 5))

        ax[0].plot(bin_centresV1, tuning_function_hdir[0, :], '.-', label='Tuning to hdir', color="cornflowerblue")
        ax[0].fill_between(bin_centresV1, tuning_function_hdir[0, :] - tf_semv1[0, :], tuning_function_hdir[0, :] + tf_semv1[0, :], alpha=0.1, color="cornflowerblue")
        ax[0].plot(bin_centresV1, tf_x_nh1_drive_by_2[0, :], '.--', label='Tuning to hdir given NH that driver is hsa', color='darkorchid')
        ax[0].fill_between(bin_centresV1, tf_x_nh1_drive_by_2[0, :] - tf_x_nh_sem12[0, :], tf_x_nh1_drive_by_2[0, :] + tf_x_nh_sem12[0, :], alpha=0.1, color='darkorchid')
        ax[0].set_xlabel('v1')
        ax[0].set_ylabel('fr')
        ax[0].legend(loc='upper right')
        ax[0].set_title("Tuning to V1", fontweight="bold")

        ax[1].plot(bin_centresV2, tuning_function_hsa[0, :], '.-', label='Tuning to hsa', color='cornflowerblue')
        ax[1].fill_between(bin_centresV2, tuning_function_hsa[0, :] - tf_semv2[0, :], tuning_function_hsa[0, :] + tf_semv2[0, :], color='cornflowerblue', alpha=0.1)
        ax[1].plot(bin_centresV2, tf_x_nh2_driven_by_1[0, :], '.--', label='Tuning to hsa given NH that driver is hdir', color='darkorchid')
        ax[1].fill_between(bin_centresV2, tf_x_nh2_driven_by_1[0, :] - tf_x_nh_sem21[0, :], tf_x_nh2_driven_by_1[0, :] + tf_x_nh_sem21[0, :], color='darkorchid', alpha=0.1)
        ax[1].set_xlabel('v2')
        ax[1].set_ylabel('fr')
        ax[1].legend(loc='upper right')
        ax[1].set_title("Tuning to V2", fontweight="bold")
        
        # To verify that the data ingested is correct, generate a polar plot
        # angles must be in radians
        ax3 = fig.add_subplot(1, 3, 3, polar=True)
        bars = ax3.bar(hdir[0], raster[0]) # index to make into 1d array
        plt.show()
        

if __name__ == '__main__':
    """The below should be set up to run as a self contained module so the user can check the module and test it works."""
    
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
    v1Object = TunED(raster, stimulusV1, Nbins, Nsamples)
    tfv1, tf_semv1, tf_s2v1, nv1, bin_centresV1 = v1Object.tuning_function(Ncells, Nbins, raster)
    v2Object = TunED(raster, stimulusV2, Nbins, Nsamples)
    tfv2, tf_semv2, tf_s2v2, nv2, bin_centresV2 = v2Object.tuning_function(Ncells, Nbins, raster)

    # Joint probability of the stimuli --------------------------------------------------------
    jointProb_stimuli, _, _ = TunED.compute_joint_prob(stimulusV1, 
                                                       stimulusV2, 
                                                       stimulusV2edges = v2Object.stimulus_bin_edges, 
                                                       stimulusV1edges = v1Object.stimulus_bin_edges,
                                                       Nbins = Nbins)
        
    # Marginalize the joint probability ------------------------------------------------------
    Pv1, Pv2 = TunED.marginalize(jointProb_stimuli)
    
    # Compute the conditional probabilities for the stimulus ---------------------------------------------------
    Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
    # Tranpose to ensure broadcasting works correctly row wise instead of column wise
    Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
    
    # ------------------------------- NULL Hypothesis tests ---------------------------------------------------------------------
    
    # apparent tuning to 'v2' given NH that cell is driven purely by v1 P(v2|v1):
    tf_x_nh2_driven_by_1, tf_x_nh_sem21 = TunED.tuning_function_null_hypothesis_PYX(tfv1, tf_s2v1, nv2, Pv1_v2)
    
    # apparent tuning to 'v1' given NH that cell is driven purely by v2 P(v1|v2):
    tf_x_nh1_drive_by_2, tf_x_nh_sem12 = TunED.tuning_function_null_hypothesis_PYX(tfv2, tf_s2v2, nv1, Pv2_v1)
    
    # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(23, 5))

    ax[0].plot(bin_centresV1, tfv1[0, :], '.-', label='Tuning to v1', color="cornflowerblue")
    ax[0].fill_between(bin_centresV1, tfv1[0, :] - tf_semv1[0, :], tfv1[0, :] + tf_semv1[0, :], alpha=0.1, color="cornflowerblue")
    ax[0].plot(bin_centresV1, tf_x_nh1_drive_by_2[0, :], '.--', label='Tuning to v1 given NH that driver is v2', color='darkorchid')
    ax[0].fill_between(bin_centresV1, tf_x_nh1_drive_by_2[0, :] - tf_x_nh_sem12[0, :], tf_x_nh1_drive_by_2[0, :] + tf_x_nh_sem12[0, :], alpha=0.1, color='darkorchid')
    ax[0].set_xlabel('v1')
    ax[0].set_ylabel('fr')
    ax[0].legend(loc='upper right')
    ax[0].set_title("Tuning to V1", fontweight="bold")

    ax[1].plot(bin_centresV2, tfv2[0, :], '.-', label='Tuning to v2', color='cornflowerblue')
    ax[1].fill_between(bin_centresV2, tfv2[0, :] - tf_semv2[0, :], tfv2[0, :] + tf_semv2[0, :], color='cornflowerblue', alpha=0.1)
    ax[1].plot(bin_centresV2, tf_x_nh2_driven_by_1[0, :], '.--', label='Tuning to v2 given NH that driver is v1', color='darkorchid')
    ax[1].fill_between(bin_centresV2, tf_x_nh2_driven_by_1[0, :] - tf_x_nh_sem21[0, :], tf_x_nh2_driven_by_1[0, :] + tf_x_nh_sem21[0, :], color='darkorchid', alpha=0.1)
    ax[1].set_xlabel('v2')
    ax[1].set_ylabel('fr')
    ax[1].legend(loc='upper right')
    ax[1].set_title("Tuning to V2", fontweight="bold")
    
    plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
    plt.show()
    
    

 # Dario recommended to not bootstrap but leaving the code in case we want to use it later
    # @staticmethod
    # def boot_strap(Ncells, hdir, hsa, raster, Nbins, Nsamples, iterations = 10, cluster_id = None, save_plot_location = None):
        
    #     # Init
    #     tuning_function_hdir = np.zeros((iterations, Nbins))
    #     tuning_function_hsa = np.zeros((iterations, Nbins))
    #     conditional_of_hdir_driven_by_hsa = np.zeros((iterations, Nbins))
    #     conditional_of_hsa_driven_by_hdir= np.zeros((iterations, Nbins))
        
    #     for sample in range(iterations):
            
    #         # Generate a bootstrap sample
    #         bootstrap_indices = np.random.choice(Nsamples, Nsamples, replace=True)
    #         bootstrap_raster = raster[:, bootstrap_indices]
    #         bootstrap_hdir = hdir[:, bootstrap_indices]
    #         bootstrap_hsa = hsa[:, bootstrap_indices]
            
    #         # Compute the tuning functions
    #         hdirObject = TunED(bootstrap_raster, bootstrap_hdir, Nbins, len(bootstrap_indices))
    #         tuning_function_hdir[sample], hdir_semv, hdir_s2, nv1, bin_centres_hdir = hdirObject.tuning_function(Ncells, Nbins, raster)
            
    #         # Compute the second tuning function
    #         hsaObject = TunED(bootstrap_raster, bootstrap_hsa, Nbins, len(bootstrap_indices)) # V2
    #         tuning_function_hsa[sample], hsa_semv, hsa_s2, nv2, bin_centres_hsa = hsaObject.tuning_function(Ncells, Nbins, raster)
            
    #         # Compute the joint probability of hdir and hsa
    #         jointProb_stimuli, _, _ = TunED.compute_joint_prob(stimulusV1 = bootstrap_hdir, 
    #                                                            stimulusV2 = bootstrap_hsa, 
    #                                                            stimulusV2edges = hsaObject.stimulus_bin_edges, 
    #                                                            stimulusV1edges = hdirObject.stimulus_bin_edges, 
    #                                                            Nbins = Nbins)
            
    #         # Compute the marginal probabilities
    #         Pv1, Pv2 = TunED.marginalize(jointProb_stimuli)
            
    #         # Compute the conditional prbabilities for the stimulus ---------------------------------------------------
    #         Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1) # P(v2|v1)
    #         # Tranpose to ensure broadcasting works correctly row wise instead of column wise
    #         Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2) # P(v1|v2)
            
    #         # Compute the conditional probability of hdir given hsa
    #         conditional_of_hdir_driven_by_hsa[sample], hdir_nh_sem = TunED.tuning_function_null_hypothesis_PYX(tuning_function_hdir[sample].reshape(1, Nbins), 
    #                                                                                                          hdir_s2, 
    #                                                                                                          nv2, 
    #                                                                                                         Pv1_v2)
            
    #         conditional_of_hsa_driven_by_hdir[sample], hsa_nh_sem = TunED.tuning_function_null_hypothesis_PYX(tuning_function_hsa[sample].reshape(1, Nbins),
    #                                                                                                           hsa_s2,
    #                                                                                                              nv1,
    #                                                                                                         Pv2_v1)
    # # FIRST PLOT ----------------------------------------------------------------------
        
    #     # variables
    #     average_hdir = np.mean(tuning_function_hdir, axis = 0)
    #     average_hsa = np.mean(tuning_function_hsa, axis = 0)
        
    #     average_hdir_nh = np.mean(conditional_of_hdir_driven_by_hsa, axis = 0)
    #     average_hsa_nh = np.mean(conditional_of_hsa_driven_by_hdir, axis = 0)
        
    #     average_hdir_sem = np.mean(hdir_semv, axis = 0)
    #     average_hsa_sem = np.mean(hsa_semv, axis = 0)
        
    #     average_hdir_nh_sem = np.mean(hdir_semv, axis = 0)
    #     average_hsa_nh_sem = np.mean(hsa_semv, axis = 0)
        
    #     # First plot the curves
    #     # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    #     fig, ax = plt.subplots(1, 3, figsize=(23, 5))
        
    #     ax[0].plot(bin_centres_hdir, average_hdir, '.-', label='Tuning to hdir', color="cornflowerblue")
    #     # ax[0].fill_between(bin_centres_hdir, average_hdir - average_hdir_sem, average_hdir + average_hdir_sem, alpha=0.1, color="cornflowerblue")
        
    #     ax[0].plot(bin_centres_hdir, average_hdir_nh, '.--', label='Tuning to hdir given NH that driver is hsa', color='darkorchid')
    #     # ax[0].fill_between(bin_centres_hdir, average_hdir_nh - average_hdir_nh_sem, average_hdir_nh + average_hdir_nh_sem, alpha=0.1, color='darkorchid')
        
    #     ax[0].set_xlabel('hdir')
    #     ax[0].set_ylabel('fr')
    #     ax[0].legend(loc='upper right')
    #     ax[0].set_title("Tuning to hdir", fontweight="bold")
        
    #     ax[1].plot(bin_centres_hsa, average_hsa, '.-', label='Tuning to hsa', color='darkorchid')
    #     # ax[1].fill_between(bin_centres_hsa, average_hsa - average_hsa_sem, average_hsa + average_hsa_sem, color='cornflowerblue', alpha=0.1)
        
    #     ax[1].plot(bin_centres_hsa, average_hsa_nh, '.--', label='Tuning to hsa given NH that driver is hdir', color="cornflowerblue")
    #     # ax[1].fill_between(bin_centres_hsa, average_hsa_nh - average_hsa_nh_sem, average_hsa_nh + average_hsa_nh_sem, color='darkorchid', alpha=0.1)
        
    #     ax[1].set_xlabel('hsa')
    #     ax[1].set_ylabel('fr')
    #     ax[1].legend(loc='upper right')
    #     ax[1].set_title("Tuning to hsa", fontweight="bold")
        
    #     # Stop plots to compute Euclidean distance
    #     # Calculate Euclidean distances between each boostrap iteration
    #     # (Samples, bin) - Calculate the norm across bins
    #     # This works because the Euclidean distance is the l2 norm, and the default value of the ord parameter in numpy.linalg.norm is 2
    #     euc_dist_hsa = np.linalg.norm(tuning_function_hsa - conditional_of_hsa_driven_by_hdir, axis=1)
    #     euc_dist_hd = np.linalg.norm(tuning_function_hdir - conditional_of_hdir_driven_by_hsa, axis=1)
        
    #     # Unit
    #     assert len(euc_dist_hsa) == iterations, "There should be a l2 norm for each bootstrap"
            
    #     # compute difference
    #     diff_distribution = euc_dist_hsa - euc_dist_hd
        
    #     # # Calculate percentiles
    #     percentile_2_5 = np.percentile(diff_distribution, 2.5)
    #     percentile_97_5 = np.percentile(diff_distribution, 97.5)
        
    #     if percentile_2_5 < 0 and percentile_97_5 < 0:
    #         outcome = "hDIR cell"
        
    #     elif percentile_2_5 > 0 and percentile_97_5 > 0:
    #         outcome = "hSA cell"
        
    #     else:
    #         outcome = "Unsure"
        
    #     # Final distribution plots
        
    #     #...
    #     ax[2].axvline(x=percentile_2_5, color='grey', linestyle='--', label='2.5 percentile')
    #     ax[2].axvline(x=percentile_97_5, color='grey', linestyle='--', label='97.5 percentile')

    #     ax[2].legend()
    #     ax[2].set_xlabel('Change of eudclid Firing Rate')
    #     ax[2].set_ylabel('Probability Density')
    #     ax[2].set_title(f'Distribution of Firing Rates for cluster {cluster_id} with a classification of: {outcome}')
        
    #     ax[2].hist(euc_dist_hsa, bins=20, density=True, color='cornflowerblue', label='dHSA')
    #     ax[2].hist(euc_dist_hd, bins=20, density=True, color='orange', label='dHD')
    #     ax[2].hist(diff_distribution, bins=20, density=True, color="black", label="dhsa - hdir")
        
    #     plt.show()
    #     # plt.savefig(f'{save_plot_location}\cluster_{cluster_id}.png')
