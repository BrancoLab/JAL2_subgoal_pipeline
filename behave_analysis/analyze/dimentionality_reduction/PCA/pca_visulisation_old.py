"""Given that the polar plots can be of shape (Neurons, Subplots, Features)
we can vectorise the data and run PCA on it to highlight trends in the data.
This is useful for identifying clusters of neurons that are similar to each other
utilising the following feature set:
-- Angle similarity
-- Magnitude of each compartment

TODO:
-- Make this work for any number of subplots
-- Include additional feature such as max firing rate
-- Remove rayleigh function and put in utils
-- Maybe move the vectorisation to postprocess
-- Check the shape of the data needed for UMAP and TSNE if same as PCA then definitely move to postprocess
-- Remove hdir cells to see if they are dominating the PCA
-- Determine the right number of clusteres
-- Remove jas plot and kmeans somewhere else
-- Smooth the firing rate

"""

import os
import math

from loguru import logger
import numpy as np
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import polars as pl

from behave_analysis.utils.rayleigh.load_rayleigh import (
    load_all_rayleigh_data,
    collect_all_rayleigh_paths,
)

from behave_analysis.utils.rayleigh.manipulate_rayleigh_df import (
    extract_compartment_values,
)

from behave_analysis.utils.rayleigh.analysis_rayleigh import angle_similarity


class PCA:
    def __init__(self, session, cluster_type, conditions, path_to_save, angles):
        """Run PCA on polar plot data.

        A class that vectorises the data used in each subplot
        of the polar plots and runs PCA on itto highlight trends
        in the data."""
        self.angles = angles
        self.session = session
        self.conditions = conditions
        self.paths = collect_all_rayleigh_paths(session, cluster_type, self.conditions)
        self.condition_data = load_all_rayleigh_data(self.paths)
        self.remove_nan_clusters()

        # Check if vectorised data exists
        try:
            x = np.load(os.path.join(path_to_save, "x.npy"))
            labels = np.load(os.path.join(path_to_save, "clu_labels.npy"))
            logger.success("Found vectorised data")
            pca_transformed_data, pca_model = self.run_pca(x)
            logger.success("Completed PCA")

        # If not vectorised data exists then vectorise it
        except FileNotFoundError:
            logger.info("No vectorised data found")
            x, labels = self.vectorise_data()
            logger.info("Running PCA")
            np.save(os.path.join(path_to_save, "x.npy"), x)
            np.save(os.path.join(path_to_save, "clu_labels.npy"), labels)
            pca_transformed_data, pca_model = self.run_pca(x)
            logger.success("Completed PCA")

        # rerun if settings says so
        # finally:
        #     if

        # K means clustering
        Kmeans = KMeans(n_clusters=12, random_state=0, n_init="auto")
        Kmeans.fit(pca_transformed_data)
        np.save(os.path.join(path_to_save, "kmeans_labels.npy"), Kmeans.labels_)
        logger.success("Completed K means clustering and saved labels")

        # Plot PCA
        self.plot_pca(pca_transformed_data, labels, pca_model, Kmeans)

        # jasmine
        self.jasimine_plot(Kmeans.labels_)

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

    def initialise_pca_matrix(self, num_neurons: int, num_features=4) -> np.array:
        """Initialise an empty data matrix of shape (neurons, features)

        Return:
        -- x: (np.array) of shape (num_neurons, num_features * num_subplots)
        """
        num_subplots = len(self.conditions) * len(self.angles)
        x = np.empty([num_neurons, num_features * num_subplots], dtype=np.float16)
        return x

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
                histogram = self.condition_data[condition][angle]["angle_firing_hist"][
                    clu_id
                ]
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

    def vectorise_data(self, num_features=4) -> tuple:
        """
        Fill a data matrix of shape (neurons, features)

        Loop through each neuron and vectorise the data for PCA.

        Return:
        -- x: (np.array) of shape (num_neurons, num_features * num_subplots)
        -- labels: (list) of cluster ids
        """
        # Initalisation
        num_neurons, clu_ids = self.extract_neurons()
        x = self.initialise_pca_matrix(num_neurons, num_features)
        labels = []  # Generate labels for color-coding

        # Loop through each neuron from top of db to bottom and insert features into X
        for n_idx, neuron_id in enumerate(clu_ids):

            iidx = 0  # Index for inserting into X
            labels.append(neuron_id)

            # Extract max firing rate
            max_firing_rate = self.find_global_max_fr(n_idx)

            for condition, _ in self.condition_data.items():
                for angle in self.condition_data[condition]:
                    subplot_df = self.condition_data[condition][angle]
                    theta = extract_compartment_values(subplot_df, "Rayleigh_theta")[
                        n_idx
                    ]
                    score = angle_similarity(theta[0], theta[1])
                    magnitude = extract_compartment_values(subplot_df, "Rayleigh")[
                        n_idx
                    ]
                    loc_max_fr = self.find_local_max_fr(
                        n_idx, max_firing_rate, subplot_df
                    )

                    # Insert into X at correct position
                    # Insert angle similarity score
                    x[n_idx, iidx] = score
                    # Insert magnitude of first compartment
                    x[n_idx, iidx + 1] = magnitude[0]
                    # Insert magnitude of second compartment
                    x[n_idx, iidx + 2] = magnitude[1]
                    # Insert local max firing rate
                    x[n_idx, iidx + 3] = loc_max_fr
                    iidx += num_features  # Move to next subplot, 3 features per subplot

        return x, labels

    def run_pca(self, x):
        """Run PCA on the vectorised data."""
        pca = decomposition.PCA(n_components=5)
        pca.fit(x)
        return pca.transform(x), pca

    def plot_pca(self, x, labels, pca_model, Kmeans):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 2)

        # 3D scatter plot - first column, spanning two rows
        ax0 = fig.add_subplot(gs[:, 0], projection="3d")
        ax0.scatter(
            x[:, 0],
            x[:, 1],
            x[:, 2],
            c=Kmeans.labels_,
            cmap=plt.cm.nipy_spectral,
            edgecolor="k",
        )
        ax0.set_xlabel("PC1")
        ax0.set_ylabel("PC2")
        ax0.set_zlabel("PC3")
        for i in range(len(x)):
            ax0.text(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                "%s" % (labels[i]),
                size=8,
                zorder=1,
                color="k",
            )

        # 2D scatter plot - first and second PCs - top row, second column
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.scatter(
            x[:, 0], x[:, 1], c=Kmeans.labels_, cmap=plt.cm.nipy_spectral, edgecolor="k"
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        for i in range(len(x)):
            ax1.text(x[i, 0], x[i, 1], "%s" % (labels[i]), size=8, zorder=1, color="k")
        # add the explained variance for each component in the title of the plot
        ax1.set_title(
            f"Explained Variance for PC 1: {pca_model.explained_variance_ratio_[0]} and PC 2: {pca_model.explained_variance_ratio_[1]}"
        )

        # 2D scatter plot - second and third PCs - bottom row, second column
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.scatter(
            x[:, 1], x[:, 2], c=Kmeans.labels_, cmap=plt.cm.nipy_spectral, edgecolor="k"
        )
        ax2.set_xlabel("PC2")
        ax2.set_ylabel("PC3")
        for i in range(len(x)):
            ax2.text(x[i, 1], x[i, 2], "%s" % (labels[i]), size=8, zorder=1, color="k")
        ax2.set_title(
            f"Explained Variance for PC 2: {pca_model.explained_variance_ratio_[1]} and PC 3: {pca_model.explained_variance_ratio_[2]}"
        )

        # 2D scatter plot - third and fourth PCs - bottom row, second column
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.scatter(
            x[:, 2], x[:, 3], c=Kmeans.labels_, cmap=plt.cm.nipy_spectral, edgecolor="k"
        )
        ax3.set_xlabel("PC3")
        ax3.set_ylabel("PC4")
        for i in range(len(x)):
            ax3.text(x[i, 2], x[i, 3], "%s" % (labels[i]), size=8, zorder=1, color="k")
        ax3.set_title(
            f"Explained Variance for PC 3: {pca_model.explained_variance_ratio_[2]} and PC 4: {pca_model.explained_variance_ratio_[3]}"
        )

        # Create a colormap normalized by the number of clusters
        n_clusters = len(np.unique(Kmeans.labels_))
        cmap = plt.cm.nipy_spectral
        norm = mcolors.Normalize(vmin=0, vmax=n_clusters - 1)

        # Function to map the cluster index to a color
        def cluster_color(i):
            return cmap(norm(i))

        # Create legend patches
        patches = [
            mpatches.Patch(color=cluster_color(i), label=f"Cluster {i}")
            for i in range(n_clusters)
        ]

        # Place the legend on the plot
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.show()

    def jasimine_plot(self, Kmeans_labels):
        # Add one for labels
        rows = len(self.conditions) + 1
        columns = len(self.angles) + 1

        first_key = list(self.condition_data.keys())[0]
        second_key = list(self.condition_data[first_key].keys())[0]

        unique_neurons = np.unique(
            self.condition_data[first_key][second_key]["clusterID"]
        )

        ticks = np.linspace(-3.14, 3.14, 3)
        tick_pos = np.linspace(0, 17, 3)

        for kmeans_cluster in np.unique(Kmeans_labels):
            # Get all neurons in cluster
            ids_in_cluster = np.where(Kmeans_labels == kmeans_cluster)

            # Create figure for each kmeans cluster
            gs = gridspec.GridSpec(
                rows,
                columns,
                bottom=0.05,
                top=0.95,
                hspace=0.5,
                wspace=0.5,
                width_ratios=[1] + [3] * (columns - 1),
                height_ratios=[1] + [3] * (rows - 1),
            )
            fig = plt.figure(figsize=(30, 45), constrained_layout=True)
            gs.tight_layout(fig, pad=0, h_pad=0, w_pad=0)
            plt.suptitle(f"K means cluster {kmeans_cluster}")

            # Add subtitles for each angle in first row
            for a_counter, a in enumerate(self.angles):
                ax = plt.subplot(gs[0, a_counter + 1])
                ax.text(0.5, 0.5, a, rotation="horizontal", va="center", ha="center")
                ax.set_axis_off()

            # Add subtitles for each condition in first column
            for c_counter, c in enumerate(self.condition_data.keys()):
                ax = plt.subplot(gs[c_counter + 1, 0])
                ax.text(0, 0.5, c, rotation="horizontal", va="center", ha="center")
                ax.set_axis_off()

            # Calculate the global maximum firing rate
            global_max = 0
            for condition in self.condition_data.values():
                for angle_data in condition.values():
                    firing_rates = np.array(
                        angle_data.filter(
                            pl.col("clusterID").is_in(
                                (unique_neurons[ids_in_cluster].tolist())
                            )
                        )["angle_firing_hist"]
                    )
                    com1 = np.array([sub_array[:18] for sub_array in firing_rates])
                    com2 = np.array([sub_array[18:] for sub_array in firing_rates])
                    current_max = max(com1.max(), com2.max())
                    if current_max > global_max:
                        global_max = current_max

            for c_counter, condition in enumerate(self.condition_data):
                counter = ((columns) * (c_counter + 1)) + 1
                con_df = self.condition_data[condition]

                # for angle in condition
                for angle_counter, angle in enumerate(self.condition_data[condition]):
                    df = con_df[angle]
                    # ax = plt.subplot(rows, columns, axs_counter + 1)

                    outer_grid = plt.subplot(gs[counter])
                    counter = counter + 1
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=outer_grid
                    )
                    ax1 = plt.subplot(inner_grid[0])
                    ax2 = plt.subplot(inner_grid[1])

                    firing_rates = np.array(
                        df.filter(
                            pl.col("clusterID").is_in(
                                (unique_neurons[ids_in_cluster].tolist())
                            )
                        )["angle_firing_hist"]
                    )

                    # Apparently all the firing rate is in one list which is aboslutey mental so need to fix that but here we go
                    # NOTE chance for bugs hard coded 18
                    # What the hell is going on
                    com1 = np.array([sub_array[:18] for sub_array in firing_rates])
                    com2 = np.array([sub_array[18:] for sub_array in firing_rates])

                    # NOTE not perfeclty centred as 18 % 2 == 0
                    # sort and center the data
                    target_index = 9

                    # Function to roll each row so that the max value is in the middle
                    def roll_row_to_middle(row):
                        max_index = np.argmax(row)  # return idx of max
                        rotation_steps = target_index - max_index
                        return np.roll(row, rotation_steps)

                    rolled1 = np.apply_along_axis(roll_row_to_middle, 1, com1)
                    rolled2 = np.apply_along_axis(roll_row_to_middle, 1, com2)

                    ax1.imshow(rolled1, aspect="auto", vmax=global_max)
                    ax2.imshow(rolled2, aspect="auto", vmax=global_max)
                    ax2.set_xticks(tick_pos)
                    ax2.set_xticklabels(ticks)

                    # label
                    ax1.set_ylabel("ID")
                    ax2.set_ylabel("ID")

            plt.show()

    def remove_nan_clusters(self):
        """Remove clusters that have NaN values"""
        
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