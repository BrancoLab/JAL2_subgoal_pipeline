"""
TunED model: disentangles the tuning of a neuron to two simultaneously recorded stimulus variables using coniditional independence tests.

Tuning function (tf): 
- Mean firing rate of a neuron to a given stimulus µ(v1) or µ(v2). See class compute_observed_tuning_function for more details.

Null hypothesis (nh): 
- We set a null hypothesis asking how much of the tuning function of one neuron to stimulus V1 can be explained by a second stimulus variable V2 and 
its interaction with V1. We compute the expected conditional E[fr(v2)|v1]. And thus the NH is that the tuning function of that neuron is purely driven 
by V2 and not V1. See class compute_null_hypothesis for more details. If the NH is true then the tuning function of that neuron should be the 
same as the expected conditional.

TODO:
- There should be some quality checks done on the ingested data because I found a spike count at 130  in one frame for one cell which is impossible so data quality is not there yet.
- Bins are not uniformly sampling with some bins empty - this might be causing unknown issues
- make coeff circular coeff 
"""

# standard libaries
import pickle

# Third party imports

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from tqdm import tqdm

# Custom imports

from behave_analysis.analyze.linshit import LinearShift
from behave_analysis.analyze.TunED.tunED_stats import TunEDModelStats
from behave_analysis.analyze.TunED.tunED_tuning_functions import (
    ComputeObservedTuningFunction,
    ComputeNullHypothesisTuningFunction,
)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.analyze.TunED.tuned_load_sig_clusters import ReturnSigClusters


class TunEdModel:
    """Conditional indepedence test"""

    def __init__(
        self, post_process_object, analyze_efizz_settings, save_dir, apply_linear_shift, cluster_type, conditions
    ):
        # Init model
        self.settings = analyze_efizz_settings
        self.post_process_object = post_process_object
        self.apply_linear_shift = apply_linear_shift
        self.cluster_type = cluster_type
        self.directory_location = save_dir + "\\" + str(self.cluster_type)
        self.conditions = conditions

        make_directory(self.directory_location)
        self.main()

    # ---------------------------- High level functions ----------------------------------------------

    def main(self):
        """Check if null distribution has been computed, if not then compute it, else load it and run TunEd model"""
        try:
            null = self.load_linear_shift_null_distribution()
            numBinsRequired4Significance = TunEDModelStats.compute_binomial_chance_distribution(null)
            logger.success("Null distribution found, computing TunEd model")
            self.accuracy_dic = self.execute_model_per_cluster(numBinsRequired4Significance)

        except FileNotFoundError:
            logger.info("No null distribution found, computing a new one")
            self.accuracy_dic = self.excute_model_per_cluster_with_linear_shift(shifted_variale="hsa")

    def execute_model_per_cluster(self, nbins4sig) -> dict:
        """
        For each neuron, across all conditions, run the TunED model to determine if the neuron is tuned to hdir or hsa.

        Returns:
        + accuracy_dic (dict): key is cluster and value is a tuple of sig such as (False, False) meaning both curves are not sig
        """
        accuracy_dic = {}  # Dict to store and return the accuracy of the model for each cluster

        # For each cluster, loop through all conditions and compute the TunED model
        for cluster in np.unique(self.post_process_object.spike_data["spike_clusters"]):
            cluster_data = self.post_process_object.video_spike_count_df.filter(pl.col("spike_clusters") == cluster)

            # Plotting params
            nrows = len(self.conditions)
            _, _ = plt.subplots(nrows, 3, figsize=(20, 5 * nrows))  # Add one index columns for the titles
            self.plot_condition_titles(
                self.conditions, nrows
            )  # Add subtitles for each condition in first column NOTE almost identical to another component could make reusable

            # Plot for each condition
            for conIdx, condition in enumerate(self.conditions):
                filtered_df = filter_video_dataframe(
                    dataframe=cluster_data, condition=condition
                )  # TODO spike video dataframe, filter function works on differnt dfs and needs to be renamed
                Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)
                if self.skip_cluster_if_no_or_too_few_spikes(cluster, filtered_df, Nsamples):
                    continue
                logger.info(
                    f"Running TunEd model for cluster {cluster} active for {Nsamples} samples and condition {condition} "
                    f"firing a total of {sum(filtered_df['spike_count'])} spikes "
                    f"with a correlation coefficient of {np.corrcoef(hdir, hsa)[0, 1]}"
                )
                hdir_tuning_object, hsa_tuning_object = self.compute_mean_firing_curves(
                    raster, hdir, hsa, Nbins, Nsamples
                )
                if self.check_if_any_bins_are_empty(hdir_tuning_object, hsa_tuning_object):
                    continue
                Pv1_v2, Pv2_v1 = self.compute_conditionals(hdir, hsa, hdir_tuning_object, hsa_tuning_object, Nbins)
                hdir_NH_object, hsa_NH_object = self.compute_NH_tuning(
                    hdir_tuning_object, hsa_tuning_object, Pv1_v2, Pv2_v1
                )
                (
                    hdir_significance,
                    hdirObservedCI,
                    hdirExpectedCI,
                    hsa_significance,
                    hsaObservedCI,
                    hsaExpectedCI,
                ) = self.compute_sig_array_between_curves_and_produce_CIs(
                    Nbins, hdir_tuning_object, hsa_NH_object, hsa_tuning_object, hdir_NH_object
                )
                hdirCI, hsaCI = self.convert_CIs_to_dict(hdirObservedCI, hdirExpectedCI, hsaObservedCI, hsaExpectedCI)
                is_hdir_sig, is_hsa_sig = self.check_if_tuning_funcs_are_sig(
                    hdir_significance, hsa_significance, num_bins_required_to_be_significant=nbins4sig
                )
                accuracy_dic[cluster] = [is_hdir_sig, is_hsa_sig]

                # Plot and save tuning functions
                self.plot_tuning_functions(
                    len(self.conditions),
                    conIdx,
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
                    hsaCI,
                )

            # Save and show if required
            plt.tight_layout()
            path = str(self.directory_location) + "\\" + f"cluster_{cluster}.png"
            plt.savefig(path)
            if self.settings.show_plots:
                plt.show()
            plt.close()

        # ------------------------Compute the number of clusters classified as hdir and hsa --------------------------------------
        self.sum_and_print_number_of_classified_cells(accuracy_dic)

        return accuracy_dic

    # ---------------------------- Linear shift functions ----------------------------------------------

    def select_significant_cluster_ids(self) -> list:
        """Select the cluster IDs that are significant in at least one compartment"""
        return ReturnSigClusters(self.post_process_object, self.settings).sig_clusters["cluster_id"].to_list()

    def excute_model_per_cluster_with_linear_shift(self, shifted_variale="hsa") -> dict:
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
        + does not take condition into things, also does not exclude shelter
        """

        # Init params for model
        fullBinomial = {}
        clusters = np.unique(self.post_process_object.spike_data["spike_clusters"])
        # clusters = np.unique(self.select_significant_cluster_ids())  # Only run the model on significant clusters 

        # For each cluster, compute the null distribution using linear shift
        for cluster in tqdm(clusters, desc="Genereating null distribution for linear shift per cluster"):
            X = self.post_process_object.video_spike_count_df.filter(pl.col("spike_clusters") == cluster)

            # Adding a check to see if the cluster has spikes or not as came across a cluster that did not exist in the synthetic data
            if len(X) == 0:
                logger.error(
                    f"Cluster {cluster} has no spikes or does not exist, skipping. Though this should not be the case."
                )
                continue

            # Get rid of clusters with less than 2k samples
            if len(X) < 2000:
                logger.warning(f"Cluster {cluster} has less than 2000 samples, skipping. Abitary cut off.")
                continue

            result = LinearShift(
                X=X,
                y=X[shifted_variale],
                stat_computation_func=self.tuned_model_user_defined_function_for_linear_shift,
                size_of_central_chunk=int(len(X) / 3),
            )
            fullBinomial[cluster] = result.pseudo_stats

        # Save the results
        savePath = str(self.directory_location) + "\\" + "linear_shift_null_distribution_binomial.pkl"
        with open(savePath, "wb") as f:
            pickle.dump(fullBinomial, f)

        return fullBinomial

    def tuned_model_user_defined_function_for_linear_shift(self, X, y):
        # Prepare data
        filtered_df = X
        filtered_df = filtered_df.with_column(y)  # Replace the NH column with the shifted NH column
        Nsamples, Nbins, hdir, hsa, raster = self.init_model_inputs(filtered_df)

        # Compute observed tuning functions
        hdir_tuning_object = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples
        )
        hsa_tuning_object = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples
        )

        # Compute probabilities
        jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(
            hdir,
            hsa,
            stimulusV2edges=hsa_tuning_object.stimulus_bin_edges,
            stimulusV1edges=hdir_tuning_object.stimulus_bin_edges,
            Nbins=Nbins,
        )
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1)  # P(v2|v1)
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2)  # P(v1|v2)

        # Compute NULL hypothesis that the driver is purely V1
        hdir_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hdir_tuning_object.tuning_func,
            observed_tuning_function_s2=hdir_tuning_object.tuning_func_s2,
            num_values_for_Px=hsa_tuning_object.n,
            conditional_Py_x=Pv1_v2,
        )

        # Compute the NULL hypothesis that the driver is purely V2
        hsa_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hsa_tuning_object.tuning_func,
            observed_tuning_function_s2=hsa_tuning_object.tuning_func_s2,
            num_values_for_Px=hdir_tuning_object.n,
            conditional_Py_x=Pv2_v1,
        )

        # Compute the significance of difference between the observed and expected tuning functions
        hdir_significance = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(
            Nbins=Nbins,
            observed_tf=hdir_tuning_object.tuning_func,
            expected_tf=hsa_NH_object.tuning_func_nh,
            observed_sem=hdir_tuning_object.tuning_func_sem,
            expected_sem=hsa_NH_object.tuning_func_nh_sem,
        )

        hsa_significance = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(
            Nbins=Nbins,
            observed_tf=hsa_tuning_object.tuning_func,
            expected_tf=hdir_NH_object.tuning_func_nh,
            observed_sem=hsa_tuning_object.tuning_func_sem,
            expected_sem=hdir_NH_object.tuning_func_nh_sem,
        )

        # have I got it the right way around?
        if y.name == "hdir":
            return np.sum(hsa_significance[0])

        elif y.name == "hsa":
            return np.sum(hdir_significance[0])

    def load_linear_shift_null_distribution(self):
        """Loads the null distribution for the linear shift method"""
        path = str(self.directory_location) + "\\" + "linear_shift_null_distribution_binomial.pkl"
        with open(path, "rb") as f:
            null_distribution = pickle.load(f)
        return null_distribution

    # ---------------------------- Plotting functions ----------------------------------------------

    def plot_tuning_functions(
        self,
        num_conditions,
        condition_indx,
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
        hsaCI,
    ):
        """For a given condition, plot and save the tuning functions for hdir and hsa"""
        col0 = plt.subplot(num_conditions, 3, condition_indx * 3 + 2)  # middle column
        col1 = plt.subplot(num_conditions, 3, condition_indx * 3 + 3)  # last column

        # Plot hdir observed vs null tuning function
        col0.set_title(f"Tuning to HDIR and the NH that driver is HSA - Sig: {is_hdir_sig}")
        col0.plot(
            hdir_tuning_object.bin_centres,
            hdir_tuning_object.tuning_func[0, :],
            ".-",
            label="Tuning to hdir",
            color="cornflowerblue",
        )
        col0.fill_between(
            hdir_tuning_object.bin_centres,
            hdir_tuning_object.tuning_func[0, :] - hdirCI["observedCI"][0],
            hdir_tuning_object.tuning_func[0, :] + hdirCI["observedCI"][0],
            alpha=0.1,
            color="cornflowerblue",
        )
        col0.plot(
            hdir_tuning_object.bin_centres,
            hsa_NH_object.tuning_func_nh[0, :],
            ".--",
            label="Tuning to hdir given NH that driver is hsa",
            color="darkorchid",
        )
        col0.fill_between(
            hdir_tuning_object.bin_centres,
            hsa_NH_object.tuning_func_nh[0, :] - hsaCI["expectedCI"][0],
            hsa_NH_object.tuning_func_nh[0, :] + hsaCI["expectedCI"][0],
            alpha=0.1,
            color="darkorchid",
        )
        col0.set_xlabel("Radians")
        col0.set_ylabel("fr")
        col0.legend(loc="upper right")

        # Plot hsa observed vs null tuning function
        col1.set_title(f"Tuning to HSA with NH that driver is hdir: {is_hsa_sig}")
        col1.plot(
            hsa_tuning_object.bin_centres,
            hsa_tuning_object.tuning_func[0, :],
            ".-",
            label="Tuning to hsa",
            color="cornflowerblue",
        )
        col1.fill_between(
            hsa_tuning_object.bin_centres,
            hsa_tuning_object.tuning_func[0, :] - hsaCI["observedCI"][0],
            hsa_tuning_object.tuning_func[0, :] + hsaCI["observedCI"][0],
            color="cornflowerblue",
            alpha=0.1,
        )
        col1.plot(
            hsa_tuning_object.bin_centres,
            hdir_NH_object.tuning_func_nh[0, :],
            ".--",
            label="Tuning to hsa given NH that driver is hdir",
            color="darkorchid",
        )
        col1.fill_between(
            hsa_tuning_object.bin_centres,
            hdir_NH_object.tuning_func_nh[0, :] - hdirCI["expectedCI"][0],
            hdir_NH_object.tuning_func_nh[0, :] + hdirCI["expectedCI"][0],
            color="darkorchid",
            alpha=0.1,
        )
        col1.set_xlabel("Radians")
        col1.set_ylabel("fr")
        col1.legend(loc="upper right")

        # Titles
        # spikes = sum(filtered_df["spike_count"])
        # plt.suptitle(
        #     f" Number of samples: {Nsamples}, V2 is the driving stimulus and V1 is the passenger stimulus."
        #     f"Cluster number {cluster}, spike number: {spikes}, corrcoeff: {np.corrcoef(hdir, hsa)[0, 1]},"
        #     f"is set 1 sig {is_hdir_sig}, is set 2 sig {is_hsa_sig}",
        #     fontweight="bold")

    def plot_condition_titles(self, conditions, nrows) -> None:
        """Plot titles and eemove the axes from the first column of subplots that act as sub titles"""
        for c_counter, c in enumerate(conditions):
            ax = plt.subplot(nrows, 3, c_counter * 3 + 1)
            ax.text(1, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=25)
            ax.set_axis_off()

    # ----------------------------- Helper functions to check data -----------------------------------

    def check_if_any_bins_are_empty(self, hdir_tuning_object, hsa_tuning_object) -> bool:
        """If bins are empty then skip the cluster, not sure if this is the best approach"""
        if hdir_tuning_object.skipCluster or hsa_tuning_object.skipCluster:
            logger.warning(
                "Skipping cluster as it has zero samples in one of the bins, this can be revisted but for now it is a conservative approach"
            )
            return True

    def skip_cluster_if_no_or_too_few_spikes(self, cluster, filtered_df, Nsamples):
        """Don't plot or process clusters with no spikes or less than 2k (abitrary) spikes"""
        if sum(filtered_df["spike_count"]) == 0:
            logger.warning("No spikes in this cluster, skipping, should not be the case")
            return True
        elif Nsamples < 2000:
            logger.warning(f"Cluster {cluster} has less than 2000 samples, skipping. Abitary cut off.")
            return True

    # ----------------------------- TunED model functions --------------------------------------------

    def init_model_inputs(self, data_df):
        """Init the inputs for the TunEd model"""
        Nsamples = len(data_df)
        Nbins = 20  # Number of bins to use to bin up the stimulus variable
        hdir = np.array(data_df["hdir"].to_numpy()).reshape(1, Nsamples)
        hsa = np.array(data_df["hsa"].to_numpy()).reshape(1, Nsamples)
        raster = np.array(data_df["spike_count"].to_numpy()).reshape(1, Nsamples)
        return Nsamples, Nbins, hdir, hsa, raster

    def compute_mean_firing_curves(self, raster, hdir, hsa, Nbins, Nsamples):
        """Compute the mean firing curves for hdir and hsa"""
        hdir_tuning_object = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hdir, Nbins=Nbins, Nsamples=Nsamples
        )
        hsa_tuning_object = ComputeObservedTuningFunction(
            spike_count_matrix=raster, stimulus_variable=hsa, Nbins=Nbins, Nsamples=Nsamples
        )
        return hdir_tuning_object, hsa_tuning_object

    def compute_conditionals(self, hdir, hsa, hdir_tuning_object, hsa_tuning_object, Nbins):
        """Compute the conditional probabilities"""
        jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(
            hdir,
            hsa,
            stimulusV2edges=hsa_tuning_object.stimulus_bin_edges,
            stimulusV1edges=hdir_tuning_object.stimulus_bin_edges,
            Nbins=Nbins,
        )
        Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
        Pv2_v1 = jointProb_stimuli / (np.ones(len(Pv2)).reshape(-1, 1) * Pv1)
        Pv1_v2 = jointProb_stimuli.T / (np.ones(len(Pv1)).reshape(-1, 1) * Pv2)  # P(v1|v2)
        return Pv1_v2, Pv2_v1

    def compute_NH_tuning(self, hdir_tuning_object, hsa_tuning_object, Pv1_v2, Pv2_v1):
        """Compute the NH tuning curves"""
        # Compute the NULL hypothesis that the driver is purely V1
        hdir_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hdir_tuning_object.tuning_func,
            observed_tuning_function_s2=hdir_tuning_object.tuning_func_s2,
            num_values_for_Px=hsa_tuning_object.n,
            conditional_Py_x=Pv1_v2,
        )

        # Compute the NULL hypothesis that the driver is purely V2
        hsa_NH_object = ComputeNullHypothesisTuningFunction(
            observed_tuning_function=hsa_tuning_object.tuning_func,
            observed_tuning_function_s2=hsa_tuning_object.tuning_func_s2,
            num_values_for_Px=hdir_tuning_object.n,
            conditional_Py_x=Pv2_v1,
        )
        return hdir_NH_object, hsa_NH_object

    # ----------------------------- Sig related functions --------------------------------------------
    def convert_CIs_to_dict(self, hdirObservedCI, hdirExpectedCI, hsaObservedCI, hsaExpectedCI):
        """Convert confidence intervals to a dictionary"""
        hdirCI = {"observedCI": hdirObservedCI, "expectedCI": hdirExpectedCI}  # Needed to plot MOE on tuning functions
        hsaCI = {"observedCI": hsaObservedCI, "expectedCI": hsaExpectedCI}
        return hdirCI, hsaCI

    def compute_sig_array_between_curves_and_produce_CIs(
        self, Nbins, hdir_tuning_object, hsa_NH_object, hsa_tuning_object, hdir_NH_object
    ):
        """Test whether the tuning functions are significantly different and produce the Confidence intervals"""
        # NOTE should probable shorten the name of these functions
        (
            hdir_significance,
            hdirObservedCI,
            hdirExpectedCI,
        ) = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(
            Nbins=Nbins,
            observed_tf=hdir_tuning_object.tuning_func,
            expected_tf=hsa_NH_object.tuning_func_nh,
            observed_sem=hdir_tuning_object.tuning_func_sem,
            expected_sem=hsa_NH_object.tuning_func_nh_sem,
        )
        (
            hsa_significance,
            hsaObservedCI,
            hsaExpectedCI,
        ) = TunEDModelStats.compute_significance_between_pairs_of_tuning_curves_set(
            Nbins=Nbins,
            observed_tf=hsa_tuning_object.tuning_func,
            expected_tf=hdir_NH_object.tuning_func_nh,
            observed_sem=hsa_tuning_object.tuning_func_sem,
            expected_sem=hdir_NH_object.tuning_func_nh_sem,
        )
        return hdir_significance, hdirObservedCI, hdirExpectedCI, hsa_significance, hsaObservedCI, hsaExpectedCI

    def sum_and_print_number_of_classified_cells(self, accuracy_dic):
        """Inform users how many cells are classified as hdir and hsa"""
        hdir_classified, hsa_classified = 0, 0
        for _, value in accuracy_dic.items():
            if value == [True, False]:
                hdir_classified += 1
            if value == [False, True]:
                hsa_classified += 1
        logger.info(f"Number of clusters classified as hdir: {hdir_classified}")
        logger.info(f"Number of clusters classified as hsa: {hsa_classified}")

    def check_if_tuning_funcs_are_sig(self, hdir_sig, hsa_sig, num_bins_required_to_be_significant) -> tuple:
        """Check if the tuning function is significantly different to the null hypothesis by summing the number of bins that are significant"""
        is_hdir_sig = False
        is_hsa_sig = False
        if np.sum(hdir_sig) > num_bins_required_to_be_significant:
            is_hdir_sig = True
            logger.success("The tuning function for head direction is significantly different to the null hypothesis")
        if np.sum(hsa_sig) > num_bins_required_to_be_significant:
            is_hsa_sig = True
            logger.success(
                "The tuning function for head shelter angle is significantly different to the null hypothesis"
            )
        return is_hdir_sig, is_hsa_sig


if __name__ == "__main__":
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.
    """

    # Init params
    Ncells = 1
    Nsamples = 100000  # Number of samples to generate, i.e. number of frames
    Nbins = 10  # Number of bins to use to bin up the stimulus variable

    # Generate stimuli
    stimulusV1 = np.random.randn(1, Nsamples)  # Driver stimulus
    stimulusV2 = stimulusV1 * 0.6 + np.random.randn(1, Nsamples)  # Passenger stimulus

    print(
        np.corrcoef(stimulusV1, stimulusV2)
    )  # Print the stimulus variables as a correlation matrix, to show variable 2 is correlated with variable 1

    # Generate spike trains from a Poisson process with a rate that depends on the stimulus V1
    frate = 0.1 * (stimulusV1[0, :] > 1.0) * stimulusV1[0, :]
    frate = np.minimum(1, frate)
    raster = np.zeros((1, Nsamples))
    raster[0, :] = np.random.poisson(frate)

    # Compute the tuning functions
    v1Object = ComputeObservedTuningFunction(raster, stimulusV1, Nbins, Nsamples)
    v2Object = ComputeObservedTuningFunction(raster, stimulusV2, Nbins, Nsamples)

    # Joint probability of the stimuli --------------------------------------------------------
    jointProb_stimuli, _, _ = TunEDModelStats.compute_joint_prob(
        stimulusV1,
        stimulusV2,
        stimulusV2edges=v2Object.stimulus_bin_edges,
        stimulusV1edges=v1Object.stimulus_bin_edges,
        Nbins=Nbins,
    )

    Pv1, Pv2 = TunEDModelStats.compute_marginal_prob(jointProb_stimuli)
    Pv2_v1 = jointProb_stimuli / (
        np.ones(len(Pv2)).reshape(-1, 1) * Pv1
    )  # P(v2|v1) # TODO write a compute conditional probability function with logic i Understand
    Pv1_v2 = jointProb_stimuli.T / (
        np.ones(len(Pv1)).reshape(-1, 1) * Pv2
    )  # P(v1|v2) # Tranpose to ensure broadcasting works correctly row wise instead of column wise

    # ------------------------------- NULL Hypothesis tests ---------------------------------------------------------------------
    V2_NH_object = ComputeNullHypothesisTuningFunction(
        v2Object.tuning_func, v2Object.tuning_func_s2, v1Object.n, Pv2_v1
    )  # E[fr(v2)|v1] given NH that cell is driven purely by V2:
    V1_NH_object = ComputeNullHypothesisTuningFunction(
        v1Object.tuning_func, v1Object.tuning_func_s2, v2Object.n, Pv1_v2
    )  # E[fr(v1)|v2] given NH that cell is driven purely by V1:

    # Plot the tuning functions and Null Hypothesis -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))

    ax[0].plot(v1Object.bin_centres, v1Object.tuning_func[0, :], ".-", label="Tuning to v1", color="cornflowerblue")
    ax[0].fill_between(
        v1Object.bin_centres,
        v1Object.tuning_func[0, :] - v1Object.tuning_func_sem[0, :],
        v1Object.tuning_func[0, :] + v1Object.tuning_func_sem[0, :],
        alpha=0.1,
        color="cornflowerblue",
    )
    ax[0].plot(
        v1Object.bin_centres,
        V2_NH_object.tuning_func_nh[0, :],
        ".--",
        label="Tuning to v1 given NH that driver is v2 E[fr(v2)|v1]",
        color="darkorchid",
    )
    ax[0].fill_between(
        v1Object.bin_centres,
        V2_NH_object.tuning_func_nh[0, :] - V2_NH_object.tuning_func_nh_sem[0, :],
        V2_NH_object.tuning_func_nh[0, :] + V2_NH_object.tuning_func_nh_sem[0, :],
        alpha=0.1,
        color="darkorchid",
    )
    ax[0].set_xlabel("v1")
    ax[0].set_ylabel("fr")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Tuning to V1", fontweight="bold")

    # ------------------------------------------- second chart ------------------------------------------------------

    ax[1].plot(v2Object.bin_centres, v2Object.tuning_func[0, :], ".-", label="Tuning to v2", color="cornflowerblue")
    ax[1].fill_between(
        v2Object.bin_centres,
        v2Object.tuning_func[0, :] - v2Object.tuning_func_sem[0, :],
        v2Object.tuning_func[0, :] + v2Object.tuning_func_sem[0, :],
        color="cornflowerblue",
        alpha=0.1,
    )
    ax[1].plot(
        v2Object.bin_centres,
        V1_NH_object.tuning_func_nh[0, :],
        ".--",
        label="Tuning to v2 given NH that driver is v1 E[fr(v1)|v2]",
        color="darkorchid",
    )
    ax[1].fill_between(
        v2Object.bin_centres,
        V1_NH_object.tuning_func_nh[0, :] - V1_NH_object.tuning_func_nh_sem[0, :],
        V1_NH_object.tuning_func_nh[0, :] + V1_NH_object.tuning_func_nh_sem[0, :],
        color="darkorchid",
        alpha=0.1,
    )
    ax[1].set_xlabel("v2")
    ax[1].set_ylabel("fr")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Tuning to V2", fontweight="bold")

    plt.suptitle(f"Number of samples: {Nsamples}, V1 is the driving stimulus and V2 is the passenger stimulus.")
    plt.show()
