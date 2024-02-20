import os
import time

import numpy as np
from loguru import logger
import polars as pl
import pickle
import matplotlib.pyplot as plt

from settings.settings_analyze_efizz import Settings_ae as Settings
from behave_analysis.analyze.regression_decoders.pytorch.working_models.oneD_output_LSTM import run_LSTM
from behave_analysis.analyze.TunED.model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import LDA
# from behave_analysis.analyze.manifold.Persistent_homology import persistent_homology
# from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh, compute_single_cluster_tuning
from behave_analysis.analyze.filtering_data.filtering_functions import extract_all_or_custom_conditions, identify_angles
from behave_analysis.analyze.classification.head_direction import classify_hdir
from behave_analysis.analyze.classification.head_shelter import classify_hsa
from behave_analysis.analyze.dimentionality_reduction.preprocessing_dim_reduce import Preprocess_for_DimReduction
from behave_analysis.analyze.dimentionality_reduction.PCA.visulisation_pca import run_pca_kmeans_plot
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.analyze.regression_decoders.sklearn_decoders.sklearn_main import sklearn_main
from behave_analysis.analyze.Rayleigh.analyze_rayleighs import plot_rayleigh_deltas
from behave_analysis.analyze.dimentionality_reduction.UMAP.umap_main import run_umap_then_hdbscan

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file.
    The purpose of this class is to make it easy to run all of the models on the same data without having to run
    the preprocessing each time. Any processing of the data should be done outside of this module.
    """

    def __init__(self, session, c_type):
        logger.info("Initializing AnalyzeEfizz")
        self.session = session
        self.dir = make_directory(os.path.join(session.base_path, session.processed_path, "models"))
        self.show_plots = Settings.show_plots
        self.settings = Settings
        self.all_conditions = extract_all_or_custom_conditions(Settings, session)
        self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")

        self.cluster_type = c_type
        assert c_type in ["synthetic", "synthetichdir", "synthetichdirhsa", "all", "good", "mua", "noise"], "Cluster type not recognised"
        assert os.path.isfile(
            os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + c_type + "_cluster_matrix.npy"
        ), "Cluster matrix file not found"
        self.frame_by_cluster_matrix = np.load(
            os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + c_type + "_cluster_matrix.npy"
        )
        self.tracking_data = open_tracking_data(self.session)

        # TODO: in postprocess save cluster Ids as separate npy file so you don't have to load in postprocess object
        self.cluster_Ids = np.load(
            str(os.path.join(self.session.base_path, self.session.processed_path) + "/" + self.cluster_type + "_cluster_Ids.npy")
        )


    def execute_models(self):
        logger.info("Executing models")

        # ----------------- Compute Rayleigh, polar plots and delta hists ------------

        if Settings.run_rayleigh:
            if not Settings.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                if Settings.learned_conditions:
                    base_path = make_directory(os.path.join(self.dir, "Rayleigh", self.cluster_type, "learned_condition"))
                else:
                    base_path = make_directory(os.path.join(self.dir, "Rayleigh", self.cluster_type, "object_condition"))
                compute_all_clusters_rayleigh(self, Settings, all_angles, self.all_conditions, base_path)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self, Settings)
                
            # Plot rayleigh deltas hists also used in dimentionality reduction so need to run rayleigh first
            # self.mangituide_deltas = plot_rayleigh_deltas(self.session, self.cluster_type)  # Analyze rayleigh deltas

        # ------------------------------ Compute TUNED --------------------------------
        if Settings.run_tunED:
            logger.info("Running TunED model")
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
            model_path = os.path.join(self.dir, "tunED")
            TunEdModel(
                video_spike_count_df=self.video_spike_count_df,
                analyze_efizz_settings=Settings,
                save_dir=model_path,
                session=self.session,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
            )
            logger.success("TunED analysis complete")

        # ------------------------------ Compute LSTM --------------------------------

        if Settings.run_LSTM:
            logger.info("Running LSTM model")
            X = self.frame_by_cluster_matrix
            Y = self.video_df["hdir"]
            run_LSTM(X, Y)
            logger.success("LSTM analysis complete")
            
            # -------- Hack to make some plots, will be removed later ------------

            # # First select all the random points columns from the video df that contain rand
            # columns_to_select = [col for col in self.video_df.columns if "head_randP" in col]

            # # Extract locations of random points
            # rand_points = self.tracking_data["randP_loc"]
            # x, y = rand_points[:, 0], rand_points[:, 1]  # Split the array into x and y coordinates
            # num_rand_points = len(rand_points)
            # assert num_rand_points == len(columns_to_select), "Number of random points does not match the number of columns"

            # # Compute R2 scores using LSTM for each random point
            # r2_scores = np.zeros(len(rand_points))
            # for i in range(len(rand_points)):
            #     Y = self.video_df[columns_to_select[i]]
            #     r2_scores[i] = run_LSTM(X, Y, verbose=False)
            #     print(f"R2 score for random point {i} is {r2_scores[i]}")
                
            # # Save the r2 scores as a numpy file
            # np.save(os.path.join(self.dir, "r2_scores.npy"), r2_scores)

            # # Plot
            # plt.figure(figsize=(10, 6))
            # sc = plt.scatter(x, y, c=r2_scores, cmap="bwr", edgecolor="k")  # 'bwr' stands for Blue-White-Red
            # plt.colorbar(sc, label="R2 Score")  # Add a colorbar to show the R2 score
            # plt.title("Random Points with R2 Scores")
            # plt.xlabel("X Coordinate")
            # plt.ylabel("Y Coordinate")
            # plt.grid(True)

            # plt.show()

        # ------------------------------ Sklearn decoder models --------------------------------
        if Settings.run_sklearn_decoders:
            sklearn_main(self.session, self.video_df, self.frame_by_cluster_matrix, cluster_labels=self.cluster_Ids)

        # ------------------------------ Compute LDA --------------------------------
        if len(Settings.run_LDA) > 0:

            LDA(self, Settings)
            logger.success('LDA analysis complete')

        # ----------------- Compute Rayleigh and polar plots -------------------------
        if Settings.run_rayleigh:
            if not Settings.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                if Settings.learned_conditions:
                    base_path = make_directory(os.path.join(self.dir, "Rayleigh", self.cluster_type, "learned_condition"))
                else:
                    base_path = make_directory(os.path.join(self.dir, "Rayleigh", self.cluster_type, "object_condition"))
                compute_all_clusters_rayleigh(self, Settings, all_angles, self.all_conditions, base_path)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self, Settings)

        # ----------------------------- Conduct Dimentionality Reduction and clustering ----------------------------------
        if Settings.run_dim_reduction:
            path_to_save = os.path.join(self.dir, "dimentionality_reduction")
            make_directory(path_to_save)

            angles = identify_angles(self.session)
            Preprocess_DimOBJ = Preprocess_for_DimReduction(
                session=self.session,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
                path_to_save=path_to_save,
                angles=angles,
                delta_between_conditions=self.mangituide_deltas,
            )
            if Settings.run_pca:
                logger.info("Running PCA model")
                run_pca_kmeans_plot(path_to_save, Preprocess_DimOBJ.x, Preprocess_DimOBJ.labels)
                logger.success("PCA analysis complete")

            if Settings.run_umap:
                angles = identify_angles(self.session)
                # Run UMAP hen HDBSCAN and return the cluster ids to the neuron ids
                cluster_ids = run_umap_then_hdbscan(Preprocess_DimOBJ.x, Preprocess_DimOBJ.labels, save_path=path_to_save)
                # TODO - Compute angle similarity for each hsbscnae cluster and then assign the cluster with the highest similarity to the hdir cluster

            # ----------------------------- Persistent Homology ----------------------------------
            # if Settings.persistent_homology:
            #     persistent_homology(self.frame_by_cluster_matrix, self.video_df)

        logger.success("All models complete")

    def classify_cells(self):
        """A function to call cell type specific classification functions

        NOTE: Work in progress"""
        hdir_cell_ids = classify_hdir(session=self.session, cluster_type=self.cluster_type)
        logger.debug(f"The hdir cell ids are: {hdir_cell_ids}")

        hsa_cell_ids = classify_hsa(
            session=self.session,
            cluster_type=self.cluster_type,
            hdir_cells=hdir_cell_ids,
        )
        logger.debug(f"The hsa cell ids are: {hsa_cell_ids}")
