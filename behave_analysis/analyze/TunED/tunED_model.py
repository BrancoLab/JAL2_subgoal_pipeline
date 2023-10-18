"""
                                                                                        Overview of the TunED model

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

# Custom imports

from behave_analysis.analyze.linshit import LinearShift
from behave_analysis.analyze.TunED.tunED_stats import TunEDModelStats
from behave_analysis.analyze.TunED.tunED_tuning_functions import ComputeObservedTuningFunction, ComputeNullHypothesisTuningFunction
from behave_analysis.utils.settings_objects import Settings_analyze_efizz as settings

# OS Imports

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from scipy.stats import norm, binom
import pickle
import os
from tqdm import tqdm
      
class TunEdModel:
    def __init__(self, 
                 analyzeEfizzObject, 
                 analyze_efizz_settings, 
                 save_location, 
                 apply_linear_shift = False):
        
        self.settings = analyze_efizz_settings
        self.analyzeEfizzObject = analyzeEfizzObject
        self.directory_location = save_location
        self.apply_linear_shift = apply_linear_shift
        self.dataDataFrame = self.filter_data_by_period() # before shelter or after shelter etc
        
        # If linear shift is not applied then check a null distribution exists
        if not self.apply_linear_shift:
            path = str(self.directory_location) + "\\" + str(settings.cluster_type) + "\\" + "linear_shift_null_distribution_binomial.pkl"
            assert os.path.exists(path), "File does not exist! You must run with lin shift first to generate the null else model has no way to compute significance"
            logger.info("Loading the null distribution for a previously computed binomial test")
            self.accuracy_dic = self.execute_model_per_cluster()
        
        # If linear shift applied, generate a new null distribution
        if self.apply_linear_shift:
            self.accuracy_dic = self.excute_model_per_cluster_with_linear_shift(shifted_variale = "hsa")
                
    def filter_data_by_period(self):
        """
        The purpose of this function is to filter the data by the period of interest but also to remove the data that is not relevant to the model such as 
        escapse periods and periods when the mouse is in the shelter.
        """
        
        if self.settings.analyze_only_the_period_before_shelter & self.settings.analyze_only_the_period_before_barrier:
            assert False, "Cannot analyze only the period before the shelter and the period before the barrier at the same time."
        
        # Filter out escape, periods when the mouse is in his house and periods when the shelter is not present
        if self.settings.analyze_only_the_period_before_shelter:
            filtered_data = self.analyzeEfizzObject.postprocessObject.video_spike_count_df.filter((self.analyzeEfizzObject.postprocessObject.video_spike_count_df["OutofshelterIdx"] == True) & 
                                                                 (self.analyzeEfizzObject.postprocessObject.video_spike_count_df["EscapePeriod"] == False) &
                                                                 (self.analyzeEfizzObject.postprocessObject.video_spike_count_df["shelter_only"] == False))
        
        # Filter out escape, and periods when the mouse is in his house, and periods when the shelter is and is not present
        if not self.settings.analyze_only_the_period_before_shelter:
            filtered_data = self.analyzeEfizzObject.postprocessObject.video_spike_count_df.filter((self.analyzeEfizzObject.postprocessObject.video_spike_count_df["OutofshelterIdx"] == True) & 
                                                                 (self.analyzeEfizzObject.postprocessObject.video_spike_count_df["EscapePeriod"] == False))
            logger.info("Analysing the whole session with escapes and periods when the mouse is in his house removed")
        
        # Filter on the period just before the barrier
        if self.settings.analyze_only_the_period_before_barrier:
            filtered_data = self.analyzeEfizzObject.postprocessObject.video_spike_count_df.filter((self.analyzeEfizzObject.postprocessObject.video_spike_count_df["barrier_present"] == False))
        
        return filtered_data
    
    def init_model_inputs(self, data_df):
        Nsamples = len(data_df)
        Nbins = 20 # Number of bins to use to bin up the stimulus variable
        hdir = np.array(data_df["hdir"].to_numpy()).reshape(1, Nsamples)
        hsa  = np.array(data_df["hsa"].to_numpy()).reshape(1, Nsamples)
        raster = np.array(data_df["spike_count"].to_numpy()).reshape(1, Nsamples)
        return Nsamples, Nbins, hdir, hsa, raster
    
    def produce_bool_of_signifiance(self, hdir_sig, hsa_sig, num_bins_required_to_be_significant) -> tuple:
        """
        The purpose of this function is to produce a boolean that indicates whether the tuning functions are significantly different or not.
        
        Args:
        + 
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
    
    def plot_and_save_tuning_functions(self, 
                                       hdir_tuning_object, 
                                       hsa_NH_object, 
                                       hsa_tuning_object, 
                                       hdir_NH_object, 
                                       filtered_df, 
                                       cluster, 
                                       hdir, 
                                       hsa, 
                                       is_hdir_sig, 
                                       is_hsa_sig, 
                                       Nsamples,
                                       hdirCI,
                                       hsaCI):
        
        # Init figure
        fig, ax = plt.subplots(1, 2, figsize=(23, 5))
        
        # Plot hdir observed vs null tuning function
        ax[0].set_title("Tuning to head direction and the NH that the driver is head shelter angle", fontweight="bold")
        ax[0].plot(hdir_tuning_object.bin_centres, hdir_tuning_object.tuning_func[0, :], '.-', label='Tuning to hdir', color="cornflowerblue")
        
        ax[0].fill_between(hdir_tuning_object.bin_centres, hdir_tuning_object.tuning_func[0, :] - hdirCI["observedCI"][0], 
                           hdir_tuning_object.tuning_func[0, :] + hdirCI["observedCI"][0], alpha=0.1, color="cornflowerblue")
        
        ax[0].plot(hdir_tuning_object.bin_centres, hsa_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to hdir given NH that driver is hsa', color='darkorchid')
        
        ax[0].fill_between(hdir_tuning_object.bin_centres, hsa_NH_object.tuning_func_nh[0, :] - hsaCI["expectedCI"][0], 
                           hsa_NH_object.tuning_func_nh[0, :] + hsaCI["expectedCI"][0], alpha=0.1, color='darkorchid')
        
        ax[0].set_xlabel('Radians')
        ax[0].set_ylabel('fr')
        ax[0].legend(loc='upper right')
        
        # Plot hsa observed vs null tuning function
        ax[1].set_title("Tuning to shelter angle and the NH that the driver is head direction", fontweight="bold")
        ax[1].plot(hsa_tuning_object.bin_centres, hsa_tuning_object.tuning_func[0, :], '.-', label='Tuning to hsa', color='cornflowerblue')
        
        ax[1].fill_between(hsa_tuning_object.bin_centres, hsa_tuning_object.tuning_func[0, :] -  hsaCI["observedCI"][0], 
                           hsa_tuning_object.tuning_func[0, :] + hsaCI["observedCI"][0], color='cornflowerblue', alpha=0.1)
        
        ax[1].plot(hsa_tuning_object.bin_centres, hdir_NH_object.tuning_func_nh[0, :], '.--', label='Tuning to hsa given NH that driver is hdir', color='darkorchid')
        
        ax[1].fill_between(hsa_tuning_object.bin_centres, hdir_NH_object.tuning_func_nh[0, :] - hdirCI["expectedCI"][0], 
                           hdir_NH_object.tuning_func_nh[0, :] + hdirCI["expectedCI"][0], color='darkorchid', alpha=0.1)
        
        ax[1].set_xlabel('Radians')
        ax[1].set_ylabel('fr')
        ax[1].legend(loc='upper right')
        
        # Titles
        spikes = sum(filtered_df["spike_count"])
        plt.suptitle(f" Number of samples: {Nsamples}, V2 is the driving stimulus and V1 is the passenger stimulus. \
                       Cluster number {cluster}, spike number: {spikes}, corrcoeff: {np.corrcoef(hdir, hsa)[0, 1]}, is set 1 sig {is_hdir_sig}, is set 2 sig {is_hsa_sig}", 
                       fontweight="bold")
        
        # Save and show if required
        path = str(self.directory_location) + "\\" + str(settings.cluster_type) + "\\" + f"_cluster_{cluster}.png"
        plt.savefig(path)
        if self.settings.show_plots:
            plt.show()
        plt.close()
    
    def execute_model_per_cluster(self):
        """
        The purpose of this function is to execute the TunEd model for each cluster in the data and thus calls all of the relevant classes and functions to do so.
        """
        
        # Load the null distribution threshold required to compute significance
        path = str(self.directory_location) + "\\" + str(settings.cluster_type) + "\\" + "linear_shift_null_distribution_binomial.pkl"
        with open(path, 'rb') as f:
            linear_shift_null_distribution_bionomial = pickle.load(f)
        numBinsRequired4Significance = TunEDModelStats.compute_binomial_chance_distribution(linear_shift_null_distribution_bionomial)
        logger.success("Loaded the null distribution for a previously computed binomial test, computing TunED model")
        
        # Init params for model
        accuracy_dic = {} # Dict to store the accuracy of the model for each cluster

        # For each cluster, compute the TunED model
        for cluster in np.unique(self.analyzeEfizzObject.postprocessObject.spike_data["spike_clusters"]):
            
            filtered_df = self.dataDataFrame.filter(pl.col("spike_clusters") == cluster)
            Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)
            logger.info(f"Running TunEd model for cluster {cluster} active for {Nsamples} samples firing a total of {sum(filtered_df['spike_count'])} spikes, with a correlation coefficient of {np.corrcoef(hdir, hsa)[0, 1]}")
            
            # Check if there are any spikes in this cluster, if not then skip
            if sum(filtered_df['spike_count']) == 0:
                logger.warning("No spikes in this cluster, skipping, should not be the case")
                continue
            
            # Get rid of clusters with less than 2k samples
            if Nsamples < 2000:
                logger.warning(f"Cluster {cluster} has less than 2000 samples, skipping. Abitary cut off.")
                continue
            
            # Compute observed tuning functions
            hdir_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hdir, Nbins = Nbins, Nsamples = Nsamples)
            hsa_tuning_object = ComputeObservedTuningFunction(spike_count_matrix = raster, stimulus_variable = hsa, Nbins = Nbins, Nsamples = Nsamples)
            
            if hdir_tuning_object.skipCluster or hsa_tuning_object.skipCluster:
                logger.warning("Skipping cluster as it has zero samples in one of the bins, this can be revisted but for now it is a conservative approach")
                continue 
            
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
            hdir_significance, hdirObservedCI, hdirExpectedCI = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins,
                                                                                                        observed_tf = hdir_tuning_object.tuning_func,
                                                                                                        expected_tf = hsa_NH_object.tuning_func_nh,
                                                                                                        observed_sem = hdir_tuning_object.tuning_func_sem,
                                                                                                        expected_sem = hsa_NH_object.tuning_func_nh_sem)
            
            hsa_significance, hsaObservedCI, hsaExpectedCI = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(Nbins = Nbins, 
                                                                                                       observed_tf = hsa_tuning_object.tuning_func,
                                                                                                       expected_tf = hdir_NH_object.tuning_func_nh,
                                                                                                       observed_sem = hsa_tuning_object.tuning_func_sem, 
                                                                                                       expected_sem = hdir_NH_object.tuning_func_nh_sem)

            # Needed to plot MOE on tuning functions
            hdirCI = {"observedCI": hdirObservedCI, "expectedCI": hdirExpectedCI}
            hsaCI = {"observedCI": hsaObservedCI, "expectedCI": hsaExpectedCI}
            
            # Produce boolean of significance
            is_hdir_sig, is_hsa_sig = self.produce_bool_of_signifiance(hdir_significance, 
                                                                       hsa_significance,
                                                                       num_bins_required_to_be_significant = numBinsRequired4Significance)
            accuracy_dic[cluster] = [is_hdir_sig, is_hsa_sig]
            
            # Plot and save tuning functions
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
                                                Nsamples,
                                                hdirCI,
                                                hsaCI)
            
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
 
        # have I got it the right way around?
        if y.name == "hdir":
            return np.sum(hsa_significance[0])
        
        elif y.name == "hsa":
            return np.sum(hdir_significance[0])
        
    def excute_model_per_cluster_with_linear_shift(self, shifted_variale = "hsa") -> dict:
        """
        A function to generate the null distribution using the linear shift method. For each cluster, the tuned model is computed on a shifted data set
        to see how many bins are significantly different between the observed and expected tuning functions. This allows us to understand how many bins
        are different by chance and thus how many bins need to be different to be significant.
        
        Args:
        + shifted_variale (str): The variable to shift, e.g. hsa or hdir
        
        Returns:
        + fullBinomial (dic): 
            Each cluster is a key, and the value is the number of bins that are significantly different between the observed and expected tuning functions for each linear shift
            so the shape of the value is (Nshifts, ) e.g fullBinomial[0] = np.array([10, 11, 10, 11, 12]) if nShifts = 5 where 10 is the number of bins that are significantly different
            
        TODO:
        + Does this function need to return anything if it is just saving the results?
        + Might need a way to make the model automatically run this first so that the null distribution is always available, instead of running it twice.
        once to generate null and then another to use the null. Might speed it up, or at least make it less manual to run the model.
        """
        
        # Init params for model
        fullBinomial = {}
        clusters = np.unique(self.analyzeEfizzObject.postprocessObject.spike_data["spike_clusters"])
        
        # For each cluster, compute the null distribution using linear shift
        for cluster in tqdm(clusters, desc="Genereating null distribution for linear shift per cluster"):
            X = self.dataDataFrame.filter(pl.col("spike_clusters") == cluster)
            
            # Adding a check to see if the cluster has spikes or not as came across a cluster that did not exist in the synthetic data
            if len(X) == 0:
                logger.error(f"Cluster {cluster} has no spikes or does not exist, skipping. Though this should not be the case.")
                continue
            
            # Get rid of clusters with less than 2k samples
            if len(X) < 2000:
                logger.warning(f"Cluster {cluster} has less than 2000 samples, skipping. Abitary cut off.")
                continue
            
            result = LinearShift(X = X, 
                                 y = X[shifted_variale], 
                                 stat_computation_func = self.tuned_model_user_defined_function_for_linear_shift,
                                 size_of_central_chunk = int(len(X) / 3))
            fullBinomial[cluster] = result.pseudo_stats
        
        # Save the results
        savePath = str(self.directory_location) + "\\" + "linear_shift_null_distribution_binomial.pkl"
        with open(savePath, 'wb') as f:
            pickle.dump(fullBinomial, f)
                        
        return fullBinomial
           
if __name__ == '__main__':
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.
    """
    
    # Init params
    Ncells = 1
    Nsamples = 100000 # Number of samples to generate, i.e. number of frames
    Nbins = 10 # Number of bins to use to bin up the stimulus variable
    
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