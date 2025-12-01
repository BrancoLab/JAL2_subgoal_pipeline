import os
import re
from collections import defaultdict, Counter
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import polars as pl
import loguru as logger
import matplotlib
from tqdm import tqdm


from settings.settings_analyze_efizz import Settings_ae
from behave_analysis.process.session import get_experiment
from behave_analysis.utils.rayleigh.load_rayleigh import collect_all_rayleigh_paths, load_all_rayleigh_data
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values, extract_firing_rates
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe

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


import os, re, pickle
from collections import Counter
from loguru import logger

# -------------------------- your imports / objects (kept minimal) -------------------------------
from settings.settings_analyze_efizz import Settings_ae as Settings
from behave_analysis.process.session import get_experiment
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe


from settings.settings_analyze_efizz import Settings_ae as Settings

from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_23apr, JAL7_30apr
from behave_analysis.database.Experiments.JAL008_ex import JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may

experiments_objects = [
    JAL6_flip7_1apr,
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_28mar,
    JAL3_25aug,
    JAL3_1sept,
    JAL3_4sept,
    JAL3_7sept,
    JAL005_8thSept,
    JAL005_21stSept,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip5_22mar,
    JAL7_flip2_12mar,
    JAL7_23apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_14may,
    JAL4_3rdSept,
    JAL4_19thSept,
    JAL4_28aug,
    JAL4_11thSept,
]

tinny_barrier = [JAL8_tiny_3may, JAL8_21may, JAL7_30apr]

# Mice groups based on session names
mice_groups = {
    "JAL6": ["JAL6_flip7_1apr", "JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_flip5_22mar", "JAL7_flip2_12mar", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_14may"],
    "JAL4": ["JAL4_3rdSept", "JAL4_19thSept", "JAL4_28aug", "JAL4_11thSept"],
    "JAL5": ["JAL005_8thSept", "JAL005_21stSept"],
}


session_NAMES = [
    "JAL6_flip7_1apr",
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL005_8thSept",
    "JAL005_21stSept",
    "JAL7_sesh8_9apr",
    "JAL7_sesh9_16apr",
    "JAL7_flip5_22mar",
    "JAL7_flip2_12mar",
    "JAL7_23apr",
    "JAL8_flip1_25apr",
    "JAL8_flip2_29apr",
    "JAL8_flip4_10may",
    "JAL8_14may",
    "JAL4_3rdSept",
    "JAL4_19thSept",
    "JAL4_28aug",
    "JAL4_11thSept",
]

# conditions = ["barrier_pre_flip", "barrier_post_flip"]
conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]
dir = make_directory(r"Z:\Jasmine_Laurence\rayleigh_analysis")
angle_keys = ["hdir_Rayleigh.arrow", "hsa_Rayleigh.arrow", "h_postflipbar_a_Rayleigh.arrow", "h_preflipbar_a_Rayleigh.arrow"]
SAVE_ROOT = r"Z:\Jasmine_Laurence\rayleigh_analysis\Top2_TunED"


class TunEdModelMod:
    """Conditional indepedence test

    Thresholding logic:
        - Rayleigh test threshold: 0.25
        - Firing rate threshold: 2.0
        - Out of shelter times only

    The model was originally designed and named to test for hdir vs hsa tuning and thus your variables will be mapped to them

    """

    def __init__(self, video_spike_count_df, session, cluster_type, conditions, v1_name="h_preflipbar_a", v2_name="h_postflipbar_a"):
        self.spike_threshold = 1000  # If a cluster has less than this number of spikes it will be skipped
        self.session = session
        self.video_spike_count_df = video_spike_count_df
        self.cluster_type = cluster_type
        self.conditions = conditions
        self.v1_name = v1_name
        self.v2_name = v2_name

    def main(self, rayleigh_data):
        """Returns the classification results for each cluster and plots the tuning functions
        for each cluster and condition if requested"""
        classification_results = self.class_ec_clu_lin_shit(rayleigh_data)
        return classification_results

    def skip_cluster_if_dud(self, cluster: int, cluster_df: pl.DataFrame, spike_thres: int) -> None:
        """Don't use clusters with no spikes or less than 2k spikes in model"""
        if len(cluster_df) == 0:
            logger.error(f"Cluster {cluster} has no spikes")
            return 1
        if len(cluster_df) < spike_thres:
            logger.error(f"Cluster {cluster} has less than {spike_thres} spikes, cut")
            return 1

    def class_ec_clu_lin_shit(self, rayleigh_data):
        """Classification of each cluster using linear shift combined with TunED

        Theory: Given a cell tuned to a single stimuli, shifting that stimuli
        should result in a significant change.Shifting the other stimuli the
        cell is NOT tuned to, should not result in a significant change.

        Thresholding logic:
        - Rayleigh test threshold: 0.25
        - Firing rate threshold: 2.0

        Returns:
        -- classification (dict): key is cluster and value is a dictionary of the
        classification results e.g =
        {"Hdir_tuned": True, "hsa_tuned": False, "mixed_tuning": False}
        """
        assert rayleigh_data is not None, "Rayleigh data must be provided"

        if 0:
            pool = PersistentPool(workers=4)  # pool for when we figure out the engineering problem
        else:
            pool = "no"  # doesn't use parallel computing

        # Remove time mouse is in the shelter
        df = self.video_spike_count_df  # shorten name
        clean_spike_df = df.filter((df["OutofshelterIdx"] == True))  # REMOVE IN SHELTER TIMES
        clusters = np.unique(clean_spike_df["spike_clusters"])
        shifted_variables = [self.v1_name, self.v2_name]
        classification = {}

        for i, cluster in enumerate(tqdm(clusters, desc="Genereating null distribution for linear shift per cluster")):

            cell_data = rayleigh_data
            reject_hsa_nh = False
            reject_hdir_nh = False
            hdir_cell = False
            hsa_cell = False
            mixed_cell = False

            # Filter and remove dud clusters because they cause issues
            x = clean_spike_df.filter(pl.col("spike_clusters") == cluster)
            if self.skip_cluster_if_dud(cluster=cluster, cluster_df=x, spike_thres=self.spike_threshold):
                continue

            for shift_var in shifted_variables:
                filter_shift_var = shift_var + "_Rayleigh.arrow"
                filt = cell_data[filter_shift_var]

                try:
                    rayleigh_out = extract_compartment_values(filt, column_name="Rayleigh")
                    _, tz_fr = extract_firing_rates(filt)

                except Exception as e:
                    logger.error(f"Could not extract rayleigh or firing rates for cluster {cluster} with error {e}")
                    continue

                try:
                    # Check rayleigh threshold is met
                    if not (rayleigh_out[i][1] > 0.25):
                        continue  # skip this cell
                    # Check firing rate threshold is met
                    if not (max(tz_fr[i]) > 2.0):
                        continue  # skip this cell

                except Exception as e:
                    logger.error(f"Could not apply thresholds for cluster {cluster} with error {e}")
                    continue

                # Compute the null distribution for each shifted variable
                result = LinearShift(
                    X=x,
                    y=x[shift_var],
                    stat_computation_func=self.user_defined_func_lin_shit,
                    size_of_central_chunk=int(len(x) / 3),
                    PPool=pool,
                )

                # Reject or accept the null hypotheses
                if shift_var == self.v1_name:
                    reject_hsa_nh = result.reject_null

                elif shift_var == self.v2_name:
                    reject_hdir_nh = result.reject_null

            # Classification logic
            if reject_hsa_nh and not reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} is tuned to {self.v1_name}")
                hdir_cell = True

            elif not reject_hsa_nh and reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} is tuned to {self.v2_name}")
                hsa_cell = True

            elif reject_hsa_nh and reject_hdir_nh:
                logger.info(f"Cluster number: {cluster} has mixed selectivity")
                mixed_cell = True

            classification[cluster] = {f"{self.v1_name}_tuned": hdir_cell, f"{self.v2_name}_tuned": hsa_cell, "mixed_tuning": mixed_cell}

        return classification

    def user_defined_func_lin_shit(self, X, y):
        """The func passed to the lin shift class"""
        # Prepare data
        filtered_df = X
        filtered_df = filtered_df.with_columns(y)  # Replace the NH column with the shifted NH column

        Nsamples = len(filtered_df)
        Nbins = 20  # Number of bins to use to bin up the stimulus variable
        hdir = np.array(filtered_df[self.v1_name].to_numpy()).reshape(1, Nsamples)
        hsa = np.array(filtered_df[self.v2_name].to_numpy()).reshape(1, Nsamples)
        raster = np.array(filtered_df["spike_count"].to_numpy()).reshape(1, Nsamples)

        # Compute observed tuning functions
        hdir_tf_obj = ComputeObservedTuningFunction(spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples)
        hsa_tf_obj = ComputeObservedTuningFunction(spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples)

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
        if y.name == self.v1_name:
            # Shift hdir, return HSA NH
            return np.sum(hdir_significance[0])

        elif y.name == self.v2_name:
            # Shift hsa, return HDIR NH
            return np.sum(hsa_significance[0])

    # ----------------------------- Helper functions to check data -----------------------------------

    def check_if_any_bins_are_empty(self, hdir_tf_obj, hsa_tf_obj) -> bool:
        """If bins are empty then skip the cluster, not sure if this is the best approach"""
        if hdir_tf_obj.skipCluster or hsa_tf_obj.skipCluster:
            logger.warning("Skipping cluster as it has zero samples in one of the bins, this can be revisted but for now it is a conservative approach")
            return True

    # ----------------------------- TunED model functions --------------------------------------------

    def init_model_inputs(self, data_df):
        """Init the inputs for the model

        NOTE - I have renamed hdir and hsa to be barrier angles"""
        n_samples = len(data_df)
        n_bins = 20  # Number of bins to use to bin up the stimulus variable
        hdir = np.array(data_df[self.v1_name].to_numpy()).reshape(1, n_samples)
        hsa = np.array(data_df[self.v2_name].to_numpy()).reshape(1, n_samples)
        raster = np.array(data_df["spike_count"].to_numpy()).reshape(1, n_samples)
        return n_samples, n_bins, hdir, hsa, raster

    def compute_mean_firing_curves(self, raster, hdir, hsa, Nbins, Nsamples):
        """Compute the mean firing curves for hdir and hsa"""
        hdir_tf_obj = ComputeObservedTuningFunction(spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples)
        hsa_tf_obj = ComputeObservedTuningFunction(spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples)
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

    def produce_CIs(self, Nbins, hdir_tf_obj, hsa_NH_object, hsa_tf_obj, hdir_NH_object):
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

    def save_classification_results(self, results) -> None:
        """Save the classification results to a pickle file"""
        file_obj = open(os.path.join(self.directory_location, "classification_results"), "wb")
        pickle.dump(results, file_obj)
        file_obj.close()


def run_tuned_using_dict_threat(v1_name="h_preflipbar_a", v2_name="h_postflipbar_a"):
    final_results = defaultdict(dict)
    for i, sesh in enumerate(experiments_objects):
        session_name = session_NAMES[i]
        logger.info(f"=== Session: {session_name} ===")
        exp = get_experiment(sesh)
        paths = collect_all_rayleigh_paths(session=exp, cluster_type="good", conditions=conditions)  # paths[condition][angles]
        condition_data = load_all_rayleigh_data(paths)  # condition_data[condition][angles]

        # Get the video and spike data
        try:
            video_and_spike_data_path = os.path.join(exp.base_path, exp.processed_path, "good_video_spike_count_df.parquet")
            df_all = pl.read_parquet(video_and_spike_data_path)
        except Exception as e:
            logger.warning(f"[{session_name}] could not find good_video_spike_count_df.parquet, skipping")
            print(e)
            continue

        # Filter to only threat zone data, assuming center is at 512 y pixels
        df_all = df_all.filter(pl.col("mouse_y_position").is_not_null() & (pl.col("mouse_y_position") < 512))

        for con in conditions:
            print(f"--- Condition: {con} ---")

            # Filter rayleigh data
            rayleigh_data = condition_data[con]
            con_df = filter_video_dataframe(dataframe=df_all, condition=con)
            model = TunEdModelMod(
                video_spike_count_df=con_df,
                session=session_name,
                cluster_type="good",
                conditions=[con],
                v1_name=v1_name,
                v2_name=v2_name,
            )
            classifcation_results = model.main(rayleigh_data=rayleigh_data)
            print(classifcation_results)
            final_results[session_name][con] = classifcation_results
    return final_results


if __name__ == "__main__":

    # A_vs_HSA = run_tuned_using_dict_threat(v1_name="h_preflipbar_a", v2_name="hsa")
    # B_vs_HSA = run_tuned_using_dict_threat(v1_name="h_postflipbar_a", v2_name="hsa")
    A_vs_B = run_tuned_using_dict_threat(v1_name="h_preflipbar_a", v2_name="h_postflipbar_a")

    # Save results
    with open(os.path.join(SAVE_ROOT, "A_vs_B_all_conditions_threat_zone.pkl"), "wb") as f:
        pickle.dump(A_vs_B, f)
