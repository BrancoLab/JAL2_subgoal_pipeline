"""
This model disentangles the tuning of a neuron to two simultaneously recorded 
stimulus variables using coniditional independence tests.

Definitions:
-- tuning function (tf) - the mean firing rate of a neuron to a given stimulus
-- null hypothesis (nh) - the tuning function of a neuron to a given stimulus is 
purely driven by a second stimulus variable

TODO:
+ There should be some quality checks done on the ingested data because I found a spike 
count at 130  in one frame for one cell which is impossible so data quality is not there yet.
+ Bins are not uniformly sampling with some bins empty - this might be causing unknown issues
+ make coeff circular coeff 
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import polars as pl
from loguru import logger
from tqdm import tqdm

from behave_analysis.analyze.stats.linshit import LinearShift
from behave_analysis.utils.PersistentPool import PersistentPool
from behave_analysis.analyze.TunED.stats import TunEDModelStats
from behave_analysis.analyze.TunED.tuning_curves import (
    ComputeObservedTuningFunction,
    ComputeNullHypothesisTuningFunction,
)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.analyze.TunED.tuned_load_sig_clusters import ReturnSigClusters


class TunEdModel:
    """Conditional indepedence test"""

    def __init__(self, video_spike_count_df, analyze_efizz_settings, session, save_dir, cluster_type, conditions):
        # Mode Params
        self.spike_threshold = 1000  # If a cluster has less than this number of spikes it will be skipped
        # This will be applied per condition during plotting, and for all conditions for classification

        # Init model
        self.session = session
        self.settings = analyze_efizz_settings
        self.video_spike_count_df = video_spike_count_df
        self.cluster_type = cluster_type
        self.directory_location = save_dir + "\\" + str(self.cluster_type)
        self.conditions = conditions

        make_directory(self.directory_location)
        self.classification_results = self.main()

    # ---------------------------- High level functions ----------------------------------------------

    def main(self, just_plots=False) -> dict:
        """Returns the classification results for each cluster and plots the tuning functions
        for each cluster and condition if requested"""
        if just_plots:
            # Just run the model for plotting and don't return the classification results
            self.run_model_4_plots()
            return None
        else:
            classification_results = self.class_ec_clu_lin_shit()
            self.save_classification_results(classification_results)
            self.run_model_4_plots()
            return classification_results

    def run_model_4_plots(self):
        """Run TunEd across clu and condition for plotting"""
        # Remove time mouse is in the shelter
        df = self.video_spike_count_df  # shorten name
        clean_spike_df = df.filter((df["OutofshelterIdx"] == True))

        for cluster in np.unique(clean_spike_df["spike_clusters"]):
            cluster_data = clean_spike_df.filter(pl.col("spike_clusters") == cluster)

            # Plotting params
            nrows = len(self.conditions)
            _, _ = plt.subplots(nrows, 3, figsize=(20, 5 * nrows))  # Add one index columns for the titles
            self.plot_condition_titles(self.conditions, nrows)

            # Plot for each condition
            for con_idx, condition in enumerate(self.conditions):
                filtered_df = filter_video_dataframe(dataframe=cluster_data, condition=condition)
                Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)

                if self.skip_cluster_if_dud(cluster=cluster, cluster_df=filtered_df, spike_thres=self.spike_threshold):
                    continue

                # Compute mean firing curves
                hdir_tf_obj, hsa_tf_obj = self.compute_mean_firing_curves(raster, hdir, hsa, Nbins, Nsamples)

                # If any of the bins are empty skip this cluster
                if self.check_if_any_bins_are_empty(hdir_tf_obj, hsa_tf_obj):
                    continue

                # compute null hypotheses
                Pv1_v2, Pv2_v1 = self.compute_conditionals(hdir, hsa, hdir_tf_obj, hsa_tf_obj, Nbins)
                hdir_NH_object, hsa_NH_object = self.compute_NH_tuning(hdir_tf_obj, hsa_tf_obj, Pv1_v2, Pv2_v1)

                # Compute CIs
                hdir_CI, hsa_CI = self.produce_CIs(Nbins, hdir_tf_obj, hsa_NH_object, hsa_tf_obj, hdir_NH_object)

                self.plot_tuning_functions(
                    condition_indx=con_idx,
                    hdir_tf_obj=hdir_tf_obj,
                    hsa_NH_object=hsa_NH_object,
                    hsa_tf_obj=hsa_tf_obj,
                    hdir_NH_object=hdir_NH_object,
                    hdirCI=hdir_CI,
                    hsaCI=hsa_CI,
                )

            # Save and show if required
            plt.tight_layout()
            path = str(self.directory_location) + "\\" + f"cluster_{cluster}.png"
            plt.savefig(path)
            if self.settings.show_plots:
                plt.show()
            plt.close()

    def select_significant_cluster_ids(self) -> list:
        """Select the cluster IDs that are significant in at least one compartment"""
        return ReturnSigClusters(self.session, self.settings).sig_clusters["cluster_id"].to_list()

    def skip_cluster_if_dud(self, cluster: int, cluster_df: pl.DataFrame, spike_thres: int) -> None:
        """Don't use clusters with no spikes or less than 2k spikes in model"""
        if len(cluster_df) == 0:
            logger.error(f"Cluster {cluster} has no spikes")
            return 1
        if len(cluster_df) < spike_thres:
            logger.error(f"Cluster {cluster} has less than {spike_thres} spikes, cut")
            return 1

    def class_ec_clu_lin_shit(self) -> dict:
        """Classification of each cluster using linear shift
        combined with TunED

        Theory: Given a cell tuned to a single stimuli, shifting that stimuli
        should result in a significant change.Shifting the other stimuli the
        cell is NOT tuned to, should not result in a significant change.

        Returns:
        -- classification (dict): key is cluster and value is a dictionary of the
        classification results e.g =
        {"Hdir_tuned": True, "hsa_tuned": False, "mixed_tuning": False}

        TODO:
        + does not compute per condition
        + does not load significant clusters
        """

        pool = PersistentPool(workers = 10)
        # Remove time mouse is in the shelter
        df = self.video_spike_count_df  # shorten name
        clean_spike_df = df.filter((df["OutofshelterIdx"] == True))

        clusters = np.unique(clean_spike_df["spike_clusters"])

        shifted_variables = ["hdir", "hsa"]
        classification = {}

        for cluster in tqdm(clusters, desc="Genereating null distribution for linear shift per cluster"):
            # Initalise vars required 4 classification results
            reject_hsa_nh = False
            reject_hdir_nh = False
            hdir_cell = False
            hsa_cell = False
            mixed_cell = False

            # Filter and remove dud clusters because they cause issues
            x = clean_spike_df.filter(pl.col("spike_clusters") == cluster)
            print(cluster)
            if self.skip_cluster_if_dud(cluster=cluster, cluster_df=x, spike_thres=self.spike_threshold):
                continue

            print('why are we here')
            for shift_var in shifted_variables:
                # Compute the null distribution for each shifted variable
                result = LinearShift(
                    X=x,
                    y=x[shift_var],
                    stat_computation_func=self.user_defined_func_lin_shit,
                    size_of_central_chunk=int(len(x) / 3),
                    PPool = 'no', # pool for when we figure out the engineering problem
                )

                # Reject or accept the null hypotheses
                if shift_var == "hdir":
                    reject_hsa_nh = result.reject_null

                elif shift_var == "hsa":
                    reject_hdir_nh = result.reject_null

            # Classification logic
            if reject_hsa_nh and not reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} is tuned to hdir")
                hdir_cell = True

            elif not reject_hsa_nh and reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} is tuned to shelter")
                hsa_cell = True

            elif reject_hsa_nh and reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} has mixed selectivity")
                mixed_cell = True

            classification[cluster] = {"Hdir_tuned": hdir_cell, "hsa_tuned": hsa_cell, "mixed_tuning": mixed_cell}

        return classification

    def user_defined_func_lin_shit(self, X, y):
        """The func passed to the lin shift class"""
        # Prepare data
        filtered_df = X
        filtered_df = filtered_df.with_columns(y)  # Replace the NH column with the shifted NH column
        Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)

        # Compute observed tuning functions
        hdir_tf_obj = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples
        )
        hsa_tf_obj = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples
        )

        # Compute probabilities
        jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(
            hdir,
            hsa,
            stimulusV2edges=hsa_tf_obj.stimulus_bin_edges,
            stimulusV1edges=hdir_tf_obj.stimulus_bin_edges,
            Nbins=Nbins,
        )
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1)  # P(v2|v1)
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2)  # P(v1|v2)

        # Compute NULL hypothesis that the driver is purely V1
        hdir_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hdir_tf_obj.tuning_func,
            observed_tuning_function_s2=hdir_tf_obj.tuning_func_s2,
            num_values_for_Px=hsa_tf_obj.n,
            conditional_Py_x=Pv1_v2,
        )

        # Compute the NULL hypothesis that the driver is purely V2
        hsa_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hsa_tf_obj.tuning_func,
            observed_tuning_function_s2=hsa_tf_obj.tuning_func_s2,
            num_values_for_Px=hdir_tf_obj.n,
            conditional_Py_x=Pv2_v1,
        )

        # Compute the significance of difference between the observed and expected tuning functions
        hdir_significance, _, _ = TunEDModelStats.compute_sig_between_curves(
            Nbins=Nbins,
            observed_tf=hdir_tf_obj.tuning_func,
            expected_tf=hsa_NH_object.tuning_func_nh,
            observed_sem=hdir_tf_obj.tuning_func_sem,
            expected_sem=hsa_NH_object.tuning_func_nh_sem,
        )

        hsa_significance, _, _ = TunEDModelStats.compute_sig_between_curves(
            Nbins=Nbins,
            observed_tf=hsa_tf_obj.tuning_func,
            expected_tf=hdir_NH_object.tuning_func_nh,
            observed_sem=hsa_tf_obj.tuning_func_sem,
            expected_sem=hdir_NH_object.tuning_func_nh_sem,
        )

        # If a cell is tuned to hdir, then shifting hdir will
        # result in a siginifcant change. And shifting hsa will
        # not result in a significant change because the cell
        # is not tuned to hsa. The opposite is true for hsa tuning.
        if y.name == "hdir":
            # Shift hdir, return HSA NH
            return np.sum(hdir_significance[0])

        elif y.name == "hsa":
            # Shift hsa, return HDIR NH
            return np.sum(hsa_significance[0])

    # ---------------------------- Plotting functions ----------------------------------------------

    def plot_tuning_functions(
        self,
        condition_indx: int,
        hdir_tf_obj: object,
        hsa_NH_object: object,
        hsa_tf_obj: object,
        hdir_NH_object: object,
        hdirCI: dict,
        hsaCI: dict,
    ) -> None:
        """For a given condition, plot and save the tuning functions for hdir and hsa"""
        col0 = plt.subplot(len(self.conditions), 3, condition_indx * 3 + 2)  # middle column
        col1 = plt.subplot(len(self.conditions), 3, condition_indx * 3 + 3)  # last column

        # Plot hdir observed vs null tuning function
        col0.set_title(f"Null: HSA is the driver", fontsize=20)
        col0.plot(
            hdir_tf_obj.bin_centres,
            hdir_tf_obj.tuning_func[0, :],
            ".-",
            label="Observed Hdir",
            color="olive",
        )
        col0.fill_between(
            hdir_tf_obj.bin_centres,
            hdir_tf_obj.tuning_func[0, :] - hdirCI["observedCI"][0],
            hdir_tf_obj.tuning_func[0, :] + hdirCI["observedCI"][0],
            alpha=0.1,
            color="olive",
        )
        col0.plot(
            hdir_tf_obj.bin_centres,
            hsa_NH_object.tuning_func_nh[0, :],
            ".--",
            label="HSA conditioned on Hdir",
            color="darkorange",
        )
        col0.fill_between(
            hdir_tf_obj.bin_centres,
            hsa_NH_object.tuning_func_nh[0, :] - hsaCI["expectedCI"][0],
            hsa_NH_object.tuning_func_nh[0, :] + hsaCI["expectedCI"][0],
            alpha=0.1,
            color="darkorange",
        )
        col0.set_xlabel("Radians", fontsize=20)
        col0.set_ylabel("Spikes (Hz)", fontsize=20)
        col0.legend(loc="upper right", fontsize=16)

        # Plot hsa observed vs null tuning function
        col1.set_title(f"Null: Hdir is the driver", fontsize=20)
        col1.plot(
            hsa_tf_obj.bin_centres,
            hsa_tf_obj.tuning_func[0, :],
            ".-",
            label="Observed HSA",
            color="olive",
        )
        col1.fill_between(
            hsa_tf_obj.bin_centres,
            hsa_tf_obj.tuning_func[0, :] - hsaCI["observedCI"][0],
            hsa_tf_obj.tuning_func[0, :] + hsaCI["observedCI"][0],
            color="olive",
            alpha=0.1,
        )
        col1.plot(
            hsa_tf_obj.bin_centres,
            hdir_NH_object.tuning_func_nh[0, :],
            ".--",
            label="Hdir conditioned on HSA",
            color="darkorange",
        )
        col1.fill_between(
            hsa_tf_obj.bin_centres,
            hdir_NH_object.tuning_func_nh[0, :] - hdirCI["expectedCI"][0],
            hdir_NH_object.tuning_func_nh[0, :] + hdirCI["expectedCI"][0],
            color="darkorange",
            alpha=0.1,
        )
        col1.set_xlabel("Radians", fontsize=20)
        col1.set_ylabel("Spikes (Hz)", fontsize=20)
        col1.legend(loc="upper right", fontsize=16)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        matplotlib.rc("xtick", labelsize=16)
        matplotlib.rc("ytick", labelsize=16)
        plt.subplots_adjust(wspace=0.01)

        return None

    def plot_condition_titles(self, conditions, nrows) -> None:
        """Plot titles and eemove the axes from the first column of subplots that act as sub titles"""
        for c_counter, c in enumerate(conditions):
            ax = plt.subplot(nrows, 3, c_counter * 3 + 1)
            ax.text(1, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=25)
            ax.set_axis_off()

    # ----------------------------- Helper functions to check data -----------------------------------

    def check_if_any_bins_are_empty(self, hdir_tf_obj, hsa_tf_obj) -> bool:
        """If bins are empty then skip the cluster, not sure if this is the best approach"""
        if hdir_tf_obj.skipCluster or hsa_tf_obj.skipCluster:
            logger.warning(
                "Skipping cluster as it has zero samples in one of the bins, this can be revisted but for now it is a conservative approach"
            )
            return True

    # ----------------------------- TunED model functions --------------------------------------------

    def init_model_inputs(self, data_df):
        """Init the inputs for the model"""
        n_samples = len(data_df)
        n_bins = 20  # Number of bins to use to bin up the stimulus variable
        hdir = np.array(data_df["hdir"].to_numpy()).reshape(1, n_samples)
        hsa = np.array(data_df["hsa"].to_numpy()).reshape(1, n_samples)
        raster = np.array(data_df["spike_count"].to_numpy()).reshape(1, n_samples)
        return n_samples, n_bins, hdir, hsa, raster

    def compute_mean_firing_curves(self, raster, hdir, hsa, Nbins, Nsamples):
        """Compute the mean firing curves for hdir and hsa"""
        hdir_tf_obj = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples
        )
        hsa_tf_obj = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples
        )
        return hdir_tf_obj, hsa_tf_obj

    def compute_conditionals(self, hdir, hsa, hdir_tf_obj, hsa_tf_obj, Nbins):
        """Compute the conditional probabilities"""
        joint_prob, _, _ = TunEDModelStats.compute_joint_prob(
            hdir,
            hsa,
            stimulusV2edges=hsa_tf_obj.stimulus_bin_edges,
            stimulusV1edges=hdir_tf_obj.stimulus_bin_edges,
            Nbins=Nbins,
        )
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(joint_prob)
        Pv2_v1 = joint_prob / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1)
        Pv1_v2 = joint_prob.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2)  # P(v1|v2)
        return Pv1_v2, Pv2_v1

    def compute_NH_tuning(self, hdir_tf_obj, hsa_tf_obj, Pv1_v2, Pv2_v1):
        """Compute the NH tuning curves"""
        # Compute the NULL hypothesis that the driver is purely V1
        hdir_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hdir_tf_obj.tuning_func,
            observed_tuning_function_s2=hdir_tf_obj.tuning_func_s2,
            num_values_for_Px=hsa_tf_obj.n,
            conditional_Py_x=Pv1_v2,
        )

        # Compute the NULL hypothesis that the driver is purely V2
        hsa_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hsa_tf_obj.tuning_func,
            observed_tuning_function_s2=hsa_tf_obj.tuning_func_s2,
            num_values_for_Px=hdir_tf_obj.n,
            conditional_Py_x=Pv2_v1,
        )
        return hdir_NH_object, hsa_NH_object

    # ----------------------------- Produce confidence intervals for plotting ------------------------
    def convert_CIs_to_dict(self, hdirObservedCI, hdirExpectedCI, hsaObservedCI, hsaExpectedCI):
        """Convert confidence intervals to a dictionary"""
        hdir_ci = {"observedCI": hdirObservedCI, "expectedCI": hdirExpectedCI}  # Needed to plot MOE on tuning functions
        hsa_ci = {"observedCI": hsaObservedCI, "expectedCI": hsaExpectedCI}
        return hdir_ci, hsa_ci

    def produce_CIs(self, Nbins, hdir_tf_obj, hsa_NH_object, hsa_tf_obj, hdir_NH_object) -> (dict, dict):
        """Produce CIs for each curve"""
        (
            _,
            hdirObservedCI,
            hdirExpectedCI,
        ) = TunEDModelStats.compute_sig_between_curves(
            Nbins=Nbins,
            observed_tf=hdir_tf_obj.tuning_func,
            expected_tf=hsa_NH_object.tuning_func_nh,
            observed_sem=hdir_tf_obj.tuning_func_sem,
            expected_sem=hsa_NH_object.tuning_func_nh_sem,
        )
        (
            _,
            hsaObservedCI,
            hsaExpectedCI,
        ) = TunEDModelStats.compute_sig_between_curves(
            Nbins=Nbins,
            observed_tf=hsa_tf_obj.tuning_func,
            expected_tf=hdir_NH_object.tuning_func_nh,
            observed_sem=hsa_tf_obj.tuning_func_sem,
            expected_sem=hdir_NH_object.tuning_func_nh_sem,
        )
        hdir_CI, hsa_CI = self.convert_CIs_to_dict(hdirObservedCI, hdirExpectedCI, hsaObservedCI, hsaExpectedCI)

        return hdir_CI, hsa_CI

    # ------------------------------- Utility functions ----------------------------------------------

    def save_classification_results(self, results: dict) -> None:
        """Save the classification results to a pickle file"""
        file_obj = open(os.path.join(self.directory_location, "classification_results"), "wb")
        pickle.dump(results, file_obj)
        file_obj.close()


# if __name__ == "__main__":
# """
# The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
# writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.
# """

# # Init params
# Ncells = 1
# Nsamples = 100000  # Number of samples to generate, i.e. number of frames
# Nbins = 10  # Number of bins to use to bin up the stimulus variable

# # Generate stimuli
# stimulusV1 = np.random.randn(1, Nsamples)  # Driver stimulus
# stimulusV2 = stimulusV1 * 0.6 + np.random.randn(1, Nsamples)  # Passenger stimulus

# print(
#     np.corrcoef(stimulusV1, stimulusV2)
# )  # Print the stimulus variables as a correlation matrix, to show variable 2 is correlated with variable 1

# # Generate spike trains from a Poisson process with a rate that depends on the stimulus V1
# frate = 0.1 * (stimulusV1[0, :] > 1.0) * stimulusV1[0, :]
# frate = np.minimum(1, frate)
# raster = np.zeros((1, Nsamples))
# raster[0, :] = np.random.poisson(frate)

# # Compute the tuning functions
# v1Object = ComputeObservedTuningFunction(raster, stimulusV1, Nbins, Nsamples)
# v2Object = ComputeObservedTuningFunction(raster, stimulusV2, Nbins, Nsamples)

# # Joint probability of the stimuli --------------------------------------------------------
# jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(
#     stimulusV1,
#     stimulusV2,
#     stimulusV2edges=v2Object.stimulus_bin_edges,
#     stimulusV1edges=v1Object.stimulus_bin_edges,
#     Nbins=Nbins,
# )

# Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
# Pv2_v1 = jointProb_stimuli / (
#     np.ones(len(Pv2)).reshape(-1, 1) * Pv1
# )  # P(v2|v1) # TODO write a compute conditional probability function with logic i Understand
# Pv1_v2 = jointProb_stimuli.T / (
#     np.ones(len(Pv1)).reshape(-1, 1) * Pv2
# )  # P(v1|v2) # Tranpose to ensure broadcasting works correctly row wise instead of column wise

# # ------------------------------- NULL Hypothesis tests ---------------------------------------------------------------------
# V2_NH_object = ComputeNullHypothesisTuningFunction(
#     v2Object.tuning_func, v2Object.tuning_func_s2, v1Object.n, Pv2_v1
# )  # E[fr(v2)|v1] given NH that cell is driven purely by V2:
# V1_NH_object = ComputeNullHypothesisTuningFunction(
#     v1Object.tuning_func, v1Object.tuning_func_s2, v2Object.n, Pv1_v2
# )  # E[fr(v1)|v2] given NH that cell is driven purely by V1:

# # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
# fig, ax = plt.subplots(1, 2, figsize=(18, 5))

# ax[0].plot(v1Object.bin_centres, v1Object.tuning_func[0, :], ".-", label="Tuning to v1", color="gold")
# ax[0].fill_between(
#     v1Object.bin_centres,
#     v1Object.tuning_func[0, :] - v1Object.tuning_func_sem[0, :],
#     v1Object.tuning_func[0, :] + v1Object.tuning_func_sem[0, :],
#     alpha=0.1,
#     color="gold",
# )
# ax[0].plot(
#     v1Object.bin_centres,
#     V2_NH_object.tuning_func_nh[0, :],
#     ".--",
#     label="Tuning to v1 given NH that driver is v2 E[fr(v2)|v1]",
#     color="darkorange",
# )
# ax[0].fill_between(
#     v1Object.bin_centres,
#     V2_NH_object.tuning_func_nh[0, :] - V2_NH_object.tuning_func_nh_sem[0, :],
#     V2_NH_object.tuning_func_nh[0, :] + V2_NH_object.tuning_func_nh_sem[0, :],
#     alpha=0.1,
#     color="darkorange",
# )
# ax[0].set_xlabel("v1")
# ax[0].set_ylabel("fr")
# ax[0].legend(loc="upper right")
# ax[0].set_title("Tuning to V1", fontweight="bold")

# # ------------------------------------------- second chart ------------------------------------------------------

# ax[1].plot(v2Object.bin_centres, v2Object.tuning_func[0, :], ".-", label="Tuning to v2", color="gold")
# ax[1].fill_between(
#     v2Object.bin_centres,
#     v2Object.tuning_func[0, :] - v2Object.tuning_func_sem[0, :],
#     v2Object.tuning_func[0, :] + v2Object.tuning_func_sem[0, :],
#     color="gold",
#     alpha=0.1,
# )
# ax[1].plot(
#     v2Object.bin_centres,
#     V1_NH_object.tuning_func_nh[0, :],
#     ".--",
#     label="Tuning to v2 given NH that driver is v1 E[fr(v1)|v2]",
#     color="darkorange",
# )
# ax[1].fill_between(
#     v2Object.bin_centres,
#     V1_NH_object.tuning_func_nh[0, :] - V1_NH_object.tuning_func_nh_sem[0, :],
#     V1_NH_object.tuning_func_nh[0, :] + V1_NH_object.tuning_func_nh_sem[0, :],
#     color="darkorange",
#     alpha=0.1,
# )
# ax[1].set_xlabel("v2")
# ax[1].set_ylabel("fr")
# ax[1].legend(loc="upper right")
# ax[1].set_title("Tuning to V2", fontweight="bold")

# plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
# plt.show()
