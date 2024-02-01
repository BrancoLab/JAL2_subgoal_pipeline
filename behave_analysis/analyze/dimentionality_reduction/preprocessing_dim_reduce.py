"""
Preprocess data for PCA and UMAP analysis. This is a class that vectorises the data used in each subplot
of the polar plots and creates a matrix of shape (neurons, features) that can be used for PCA.
If other features are to be added to the PCA analysis then they should be added here.

Current features used are:
    - Angle similarity score
    - Magnitude of first compartment
    - Magnitude of second compartment
    # - Local max firing rate - this one was coded up but removed in the end
    - Delta between conditions in rayleigh magnitude
    
TODO = refactor me im dieing of ugly code
- make into a genral step before PCA and UMAP
- clean functions
- add doc strings
- add type hints
- add unit tests
"""

import os
import math

from loguru import logger
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from behave_analysis.analyze.filtering_data.filtering_functions import identify_angles
from behave_analysis.utils.rayleigh.load_rayleigh import load_all_rayleigh_data, collect_all_rayleigh_paths
from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import extract_compartment_values
from behave_analysis.utils.rayleigh.analysis_rayleigh import angle_similarity
from settings.settings_analyze_efizz import Settings_ae as settings


class Preprocess_for_DimReduction:
    """Preprocess data for dimentionality reduction

    Creates two main attributes:
    -- x: (np.array) of shape (num_neurons, num_features * num_subplots)
    -- labels: (list) of cluster ids

    Features used in the PCA analysis:
    - Angle similarity score
    - Magnitude of first compartment
    - Magnitude of second compartment

    Thus num features is 3 but then we mulitple that by the number of subplots (conditions * angles)
    """

    def __init__(self, session, cluster_type, conditions, path_to_save, angles, delta_between_conditions):
        """Vectorises the data used in each subplot
        of the polar plots"""
        logger.info("Preprocessing data for PCA")
        self.angles = angles
        self.session = session
        self.delta_between_conditions = delta_between_conditions
        self.num_features = 3  # lesve this number its hard coded
        self.conditions = conditions
        self.paths = collect_all_rayleigh_paths(session, cluster_type, self.conditions)
        self.condition_data = load_all_rayleigh_data(self.paths)
        self.remove_nan_clusters()
        self.plot_histograms_of_max_firing_rates(path_to_save)
        x, labels = self.run_load_create_save(path_to_save)

        # Attach main attributes
        self.x = x
        self.labels = labels

    def run_load_create_save(self, path_to_save) -> tuple:
        """Load data, create it or re-run it depending on settings."""
        # Check if vectorised data exists
        try:
            if settings.redo_pca_preprocessing:
                logger.info("Re-running PCA preprocessing")
                x, labels = self.vectorise_data()
                extended_x = self.add_otherdelta_features_to_x(x, self.delta_between_conditions)
                self.save_labels_and_x(path_to_save, extended_x, labels)
                logger.success("Completed preprocessing for PCA")
            elif not settings.redo_pca_preprocessing:
                x = np.load(os.path.join(path_to_save, "x.npy"))
                labels = np.load(os.path.join(path_to_save, "clu_labels.npy"))
                logger.success("Found vectorised data")

        # If not vectorised data exists then vectorise it
        except FileNotFoundError:
            logger.info("No vectorised data found")
            logger.info("Running PCA preprocessing")
            x, labels = self.vectorise_data()
            extended_x = self.add_otherdelta_features_to_x(x, self.delta_between_conditions)
            self.save_labels_and_x(path_to_save, extended_x, labels)
            logger.success("Completed preprocessing for PCA")
        return x, labels

    def save_labels_and_x(self, path_to_save, x, labels) -> None:
        """Save labels and x to a specified path"""
        np.save(os.path.join(path_to_save, "x.npy"), x)
        np.save(os.path.join(path_to_save, "clu_labels.npy"), labels)

    def extract_neurons(self) -> tuple:
        """Extract the number of neurons from the data.

        NOTE: Assumes that the number of neurons is
        the same across all conditions and angles. I.E that
        the cluster ids are the same across all conditions and angles.

        Return:
        -- num_neurons: (int) number of neurons in the data
        -- clu_ids: (np.array) of cluster ids
        """
        first_key = list(self.condition_data.keys())[0]
        second_key = list(self.condition_data[first_key].keys())[0]
        clu_ids = np.unique(self.condition_data[first_key][second_key]["clusterID"])
        num_neurons = len(clu_ids)
        return num_neurons, clu_ids

    def initialise_pca_matrix(self, num_neurons: int) -> np.array:
        """Initialise an empty data matrix of shape (neurons, features)

        Return:
        -- x: (np.array) of shape (num_neurons, num_features * num_subplots)
        """
        num_subplots = len(self.conditions) * len(self.angles)
        x = np.empty([num_neurons, self.num_features * num_subplots], dtype=np.float16)
        return x

    def find_local_max_fr_non_normalised(self, n_idx, subplot_data, compartment: str) -> float:
        """Extract and don't normalise the local max firing rate for a single subplot for a single bin

        This function finds the maximum firing rate for a single bin and does not normalise it.
        The arguments focus this function on a single condition for a single neuron.

        Because Jasmine is insane both comaprtments are in a single array and the first 18 bins
        refer to the first compartment and the last 18 bins refer to the second compartment.

        Arguments:
        -- n_idx: (int) cluster id
        -- subplot_data: (pl.df) Polars dataframe of a single subplot"""
        if compartment == "shelter":
            return subplot_data["angle_firing_hist"][n_idx][0:18].max()
        if compartment == "threat":
            return subplot_data["angle_firing_hist"][n_idx][18:-1].max()

    def plot_histograms_of_max_firing_rates(self, path_to_save) -> None:
        """Plot histograms of max firing rates for each compartment

        Given that the max firing rates will be zscored and then each comaprtment
        difference will be computed. This function plots the distribution of max firing
        to ensure the distributions are similar. This is a sanity check.

        Saves the plot to the path_to_save. Returns nothing"""
        shelter_max_frs = self.loop_through_find_local_max_fr_non_normalised(compartment="shelter")
        threat_max_frs = self.loop_through_find_local_max_fr_non_normalised(compartment="threat")
        shelter_all_values = np.array(list(shelter_max_frs.values())).T
        threat_all_values = np.array(list(threat_max_frs.values())).T
        plt.hist(shelter_all_values.flatten(), bins=100, alpha=0.5, label="Shelter")
        plt.hist(threat_all_values.flatten(), bins=100, alpha=0.5, label="Threat")
        plt.legend()
        plt.xlabel("Max firing rate (Hz)")
        plt.ylabel("Count of neurons in a subplot")
        plt.title("Distribution of max firing rates for each compartment")
        path = os.path.join(path_to_save, "max_firing_rates_histograms.png")
        plt.savefig(path)
        plt.close()

    def vectorise_data(self) -> tuple:
        """
        Fill a data matrix of shape (neurons, features)

        Loop through each neuron and vectorise the data for PCA.

        Return:
        -- x: (np.array) of shape (num_neurons, num_features * num_subplots)
        -- labels: (list) of cluster ids
        """
        # Initalisation
        num_neurons, clu_ids = self.extract_neurons()
        x = self.initialise_pca_matrix(num_neurons)
        labels = []  # Generate labels for color-coding

        countt = 0
        # Loop through each neuron from top of db to bottom and insert features into X
        for n_idx, neuron_id in enumerate(clu_ids):
            countt += 1
            iidx = 0  # Index for inserting into X
            labels.append(neuron_id)

            for condition, _ in self.condition_data.items():
                for angle in self.condition_data[condition]:
                    subplot_df = self.condition_data[condition][angle]

                    theta = extract_compartment_values(subplot_df, "Rayleigh_theta")[n_idx]
                    score = angle_similarity(theta[0], theta[1])
                    magnitude = extract_compartment_values(subplot_df, "Rayleigh")[n_idx]

                    # Insert into X at correct position
                    x[n_idx, iidx] = score  # Insert angle similarity score
                    x[n_idx, iidx + 1] = magnitude[0]  # Insert magnitude of first compartment
                    x[n_idx, iidx + 2] = magnitude[1]  # Insert magnitude of second compartment

                    iidx += self.num_features  # Move to next subplot, 3 features per subplot

        return x, labels

    def add_otherdelta_features_to_x(self, x, deltas_between_conditions):
        """Add delta in rayleigh magnitude between conditions to the PCA matrix

        As a quick test to see if the delta between conditions is useful, I will append
        the values created for each neuron between the conditions pairs to the PCA matrix
        and thus the feature matrix will be something like
        -- feautres = num features * subplots + num delta between conditions

        """
        angles = identify_angles(self.session)
        neuron_feature_dic = {}
        for neuron in range(x.shape[0]):
            feautres = []
            for condition_pair, value in deltas_between_conditions.items():
                for angle in angles:
                    for compartment in ["one_delta", "two_delta"]:
                        feautres.append(value[angle][compartment][neuron])
            neuron_feature_dic[neuron] = feautres

        print("Lenth of new feautres is: ", len(feautres))
        print("Length of old features is: ", x.shape[1])
        old_feature_num = x.shape[1]

        # Create a new matrix for the addition data
        new_x = np.empty([x.shape[0], len(feautres)], dtype=np.float16)

        # Populate the new matrix
        for neuron in range(x.shape[0]):
            new_x[neuron] = neuron_feature_dic[neuron]

        # Concatenate the two matrices
        combined_x = np.concatenate((x, new_x), axis=1)

        # Check that the new x matrix is the correct shape
        assert combined_x.shape[1] == len(feautres) + old_feature_num, "The new x matrix is not the correct shape"

        # Now using the delta fr between compartments add that matrix
        # remove this line if you want to remove the delta fr between compartments
        # combined_combined_x = np.concatenate((combined_x, self.delta_fr_compartments), axis=1)

        return combined_x

    def remove_nan_clusters(self):
        """Remove clusters that have NaN values

        NOTE: Definitely should not be doing this here but quick fix for now
        this should be sorted outside of this class more rigourously
        """

        def find_nan_subarrays(arrays):
            nan_indices = []
            for i, sub_array in enumerate(arrays):
                if any(math.isnan(x) for x in sub_array):
                    nan_indices.append(i)
            return nan_indices

        clu_ids_with_no_firing = []
        for condition, _ in self.condition_data.items():
            for angle in self.condition_data[condition]:
                data = np.asarray(self.condition_data[condition][angle]["Rayleigh"])
                clu_ids_with_no_firing.append(find_nan_subarrays(data))

        cluster_ids_to_remove = np.unique([clu for sublist in clu_ids_with_no_firing for clu in sublist])

        # loop through each condition and angle and remove the cluster ids
        for condition, _ in self.condition_data.items():
            for angle in self.condition_data[condition]:
                df = self.condition_data[condition][angle].to_pandas()
                df = df.drop(cluster_ids_to_remove)
                self.condition_data[condition][angle] = pl.from_pandas(df)

    def loop_through_find_local_max_fr_non_normalised(self, compartment) -> dict:
        """Extract max firing rates for all conditions, all neurons, all angles

        Loops through all conditions, all angles and all neurons and extracts the max firing rate
        for a single bin. This is not normalised.

        Returns:
        -- dictionary: (dict) of cluster ids as keys and a list of max firing rates as values
            where each value is some combination of an angle and a condition
        """

        # Initalisation
        _, clu_ids = self.extract_neurons()

        dictionary = {}

        for n_idx, neuron_id in enumerate(clu_ids):
            list_of_firing_rates = []
            for condition, _ in self.condition_data.items():
                for angle in self.condition_data[condition]:
                    subplot_df = self.condition_data[condition][angle]
                    max_fr = self.find_local_max_fr_non_normalised(n_idx, subplot_df, compartment)
                    list_of_firing_rates.append(max_fr)
            dictionary[neuron_id] = list_of_firing_rates

        assert np.array(list(dictionary.keys())).all() == np.asarray(clu_ids).all(), "The dictionary keys are not the same as the cluster ids"

        return dictionary

    # ________ NOT USED FEATURE FUNCTIONS __________
    # Added the bellow features but they were not used in end as made worse performance
    # not sufficient testing so leaving here for now

    def zscore_max_firing_rates(self) -> dict:
        """For both compartments zscore the max firing rates

        Returns:
        -- zscored_max_first: (nested dict)
            keys are compartment names
                keys are cluster ids
                    values are zscored max firing rates for each configuration of angle and condition
        """
        shelter_max_frs = self.loop_through_find_local_max_fr_non_normalised(compartment="shelter")
        threat_max_frs = self.loop_through_find_local_max_fr_non_normalised(compartment="threat")

        # Prepare the data for scaling
        shelter_all_values = np.array(list(shelter_max_frs.values())).T
        threat_all_values = np.array(list(threat_max_frs.values())).T

        # Apply StandardScaler
        scaler = StandardScaler()
        scaled_values_shelter = scaler.fit_transform(shelter_all_values)
        scaled_values_threat = scaler.fit_transform(threat_all_values)

        # Reassign the transformed data back to the dictionary
        zscore_shelter_max_frs = {}
        zscore_threat_max_frs = {}
        for i, neuron in enumerate(shelter_max_frs.keys()):
            zscore_shelter_max_frs[neuron] = scaled_values_shelter[:, i].tolist()
            zscore_threat_max_frs[neuron] = scaled_values_threat[:, i].tolist()

        # create a dictionary of dictionaries
        zscored_max_frs = {"shelter": zscore_shelter_max_frs, "threat": zscore_threat_max_frs}
        return zscored_max_frs

    def turn_zscore_fr_into_delta_matrix(self) -> np.array:
        """Create delta matrix from zscored max firing rates between compartments

        Converts the zscored max firing rates into a matrix to be
        concatenated with the other features for dimentionality reduction"""

        zscored_max_frs = self.zscore_max_firing_rates()

        # Initalisation
        num_neurons, clu_ids = self.extract_neurons()

        # Compute delta of zscore between compartment firing rates
        # key for second shape length is arbitrary as they are the same
        matrix1 = np.empty([num_neurons, len(zscored_max_frs["shelter"][clu_ids[0]])], dtype=np.float16)
        matrix2 = np.empty([num_neurons, len(zscored_max_frs["threat"][clu_ids[0]])], dtype=np.float16)
        for neuron in range(num_neurons):
            matrix1[neuron] = zscored_max_frs["shelter"][clu_ids[neuron]]
            matrix2[neuron] = zscored_max_frs["threat"][clu_ids[neuron]]
        delta_matrix = matrix1 - matrix2

        return delta_matrix

    def find_global_max_fr(self, clu_id) -> float:
        """Find the maximum single bin fr for a single neuron

        Loop through all conditions and angles to find the
        maximum firing rate for a single neuron.

        Returns:
            max_fr: (float) Across all conditions and angles for a single neuron
            what is the maximum firing rate in a single bin.
        """
        max_fr = 0
        for condition, _ in self.condition_data.items():
            for angle in self.condition_data[condition]:
                histogram = self.condition_data[condition][angle]["angle_firing_hist"][clu_id]
                if np.amax(np.asarray(histogram)) > max_fr:
                    max_fr = np.amax(np.asarray(histogram))

        assert max_fr > 0, "Max firing rate is 0 which is not possible"
        return max_fr

    def find_local_max_fr(self, n_idx, max_fr, subplot) -> float:
        """Extract and normalise the local max firing rate for a single subplot

        Args:
            n_idx (int): index of neuron position in the table
            max_fr (float): Global max firing rate for a single neuron across all conditions and angles
            subplot (pl.df): Polars dataframe of a single subplot

        Returns:
            float: (float) normalised local max firing rate for a single subplot
        """
        data = subplot["angle_firing_hist"][n_idx]
        fr = np.amax(np.asarray(data))
        norm_fr = fr / max_fr
        return norm_fr
