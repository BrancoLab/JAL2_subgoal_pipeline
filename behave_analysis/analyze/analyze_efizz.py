import os
import time
from pathlib import Path

import numpy as np
from loguru import logger
import polars as pl
import pickle
import matplotlib.pyplot as plt

from settings.settings_analyze_efizz import Settings_ae as Settings
from behave_analysis.analyze.regression_decoders.pytorch.working_models.oneD_output_LSTM import run_LSTM
from behave_analysis.analyze.TunED.model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import LDA
from behave_analysis.analyze.EscapePattern.ComputeEscapeTuning import ComputeEscapeTuning

# from behave_analysis.analyze.manifold.Persistent_homology import persistent_homology
# from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh
# from behave_analysis.analyze.single_trial.predict_future import select_neural_activity_chunk, explore_neural_activity_over_time
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
from behave_analysis.analyze.single_trial.single_trial_regression import SingleTrialRegression
from behave_analysis.analyze.single_trial.preprocess_regression import PreprocessSingleTrialRegression


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
        self.cluster_type = c_type
        assert c_type in ["synthetic", "synthetichdir", "synthetichdirhsa", "all", "good", "mua", "noise"], "Cluster type not recognised"    

    def load_data(self, analysis_name):
        """A function to load data needed for each analysis type."""

        if analysis_name == 'tunED':
            # Load the video spike count data
            try:
                video_and_spike_data_path = os.path.join(self.session.base_path, self.session.processed_path, "good_video_spike_count_df.parquet")
                self.video_and_spike_data = pl.read_parquet(video_and_spike_data_path)
            except FileNotFoundError:
                logger.warning("Video and spike data not found. Eiter the file name is incorrect or the file does not exist (I did remove .parquet)")
                video_and_spike_data_path = os.path.join(self.session.base_path, self.session.processed_path, "good_video_spike_count_df")
                self.video_and_spike_data = pl.read_parquet(video_and_spike_data_path)
        
        # load video_df, frame by cluster matrix and cluster_Ids
        if analysis_name in ['LDA', 'sklearn', 'LSTM', 'rayleigh', 'EscapePattern', 'PCA', 'UMAP', 'single_trial']:
            # load behavioral data
            self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")
            # load firing rate matrix
            assert os.path.isfile(
                os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + self.cluster_type + "_cluster_matrix.npy"
            ), "Cluster matrix file not found"
            self.frame_by_cluster_matrix = np.load(
                os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + self.cluster_type + "_cluster_matrix.npy"
            )
            # load cluster Ids
            try:
                self.cluster_Ids = np.load(
                    str(os.path.join(self.session.base_path, self.session.processed_path) + "/" + self.cluster_type + "_cluster_Ids.npy")
                )
            except FileNotFoundError:
                logger.warning("Cluster Ids not found")

        # load tracking data (DLC output)
        if analysis_name == 'LDA':
            self.tracking_data = open_tracking_data(self.session)

        # Load the homings object
        if analysis_name == 'single_trial' or analysis_name == 'EscapePattern':
            try:
                homing_path = os.path.join(self.session.base_path, self.session.processed_path, "homings", "homings_obj.pkl")
                with open(homing_path, "rb") as f:
                    self.homings_object = pickle.load(f)

                escape_path = os.path.join(self.session.base_path, self.session.processed_path, "escapes", "escapes_obj.pkl")
                with open(escape_path, "rb") as f:
                    self.escape_object = pickle.load(f)
            except FileNotFoundError:
                logger.warning("Homings or escapes object not found")
            

    def execute_models(self, analysis_name=None):
        """A function to call all of the analysis models set in the settings file."""
        logger.info("Executing models")

        # ----------------- Conduct single trial analysis ----------------------------

        if analysis_name == 'single_trial':
            logger.info("Running single trial analysis")
            single_trial_save_path = Path(make_directory(os.path.join(self.dir, "single_trial")))

            # Select velocity data
            velocity_data = self.tracking_data["avg_Velocity"]  # Velocity len one less than video df because its between frames
            
            condition = "barrier_pre_flip"

            pp_single_trial_obj = PreprocessSingleTrialRegression(
                video_df=self.video_df,
                homings_obj=self.homings_object,
                frame_by_cluster_matrix=self.frame_by_cluster_matrix,
                save_path=single_trial_save_path,
                velocity_data=velocity_data,
                similar_homings=False,
                barrier_location=self.tracking_data["barrier_loc"],
                shelter_location=self.tracking_data["shelter_loc"],
                escape_object=self.escape_object,
                remove_escapes=False,
                condition = condition
            )

            # # Save the preprocessed regression object
            # with open(single_trial_save_path / "pp_single_trial_obj.pkl", "wb") as f:
            #     pickle.dump(pp_single_trial_obj, f)
            #     logger.success("Preprocessed single trial object saved, ready for analysis")

            # path = os.path.join(self.session.base_path, self.session.processed_path, "models", "single_trial", "pp_single_trial_obj.pkl")
            # with open(path, "rb") as hf:
            #     pp_single_trial_obj = pickle.load(hf)
            
            SingleTrialRegression(
                design_matrix=pp_single_trial_obj.design_matrix,
                save_path=single_trial_save_path,
                session=self.session,
                dependents_df=pp_single_trial_obj.targets_df,
                tracking_data=self.tracking_data,
                homing_list=pp_single_trial_obj.homing_list,
                spike_homing_list=pp_single_trial_obj.spike_data_per_homing,
                condition_per_homing=pp_single_trial_obj.condition_per_homing,
                cluster_ids=self.cluster_Ids,
                initial_directions=pp_single_trial_obj.initial_directions,
                conversion_from_left_right_to_pre_post_flip=pp_single_trial_obj.convert_left_right_to_pre_post_flip,
                condition= condition
            )

        #  ----------------- Compute Rayleigh, polar plots and delta hists ------------

        if analysis_name == 'rayleigh':
            logger.info(f"Compute Rayleigh on {self.cluster_type} data")
            all_angles = identify_angles(self.session)
            compute_all_clusters_rayleigh(self, all_angles)

        # ------------------------------ Compute TUNED --------------------------------
        if analysis_name == 'tunED':
            logger.info("Running TunED model")
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
            model_path = os.path.join(self.dir, "tunED")
            TunEdModel(
                video_spike_count_df=self.video_and_spike_data,
                analyze_efizz_settings=Settings,
                save_dir=model_path,
                session=self.session,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
            )
            logger.success("TunED analysis complete")

        # ------------------------------ Compute LDA --------------------------------
        if analysis_name == 'LDA':
            LDA(self)
            logger.success("LDA analysis complete")

# ------------------------------ Compute LDA --------------------------------
        if analysis_name == 'EscapePattern':
            logger.info("Running Escape Pattern Tuning model")

            ComputeEscapeTuning(aefizz = self).extract_data(aefizz = self)
            ComputeEscapeTuning(aefizz = self).compute_tuning_curves(aefizz = self)
            ComputeEscapeTuning(aefizz = self).compute_statistical_significance(aefizz = self)

            logger.success("Escape Pattern Tuning analysis complete")

        # ----------------------------- Conduct Dimentionality Reduction and clustering ----------------------------------
        if analysis_name == 'PCA' or analysis_name == 'UMAP':

            raise NotImplementedError("Dimentionality reduction code is being updated and is not currently available")
            # TODO: Dim red needs to either run or load rayleigh to compute mangitiude deltas
            # logger.info("Running Dimentionality Reduction models")
            # path_to_save = os.path.join(self.dir, "dimentionality_reduction")
            # make_directory(path_to_save)

            # # Plot rayleigh deltas hists also used in dimentionality reduction so need to run rayleigh first
            # # self.mangituide_deltas = plot_rayleigh_deltas(self.session, self.cluster_type)  # Analyze rayleigh deltas

            # angles = identify_angles(self.session)
            # Preprocess_DimOBJ = Preprocess_for_DimReduction(
            #     session=self.session,
            #     cluster_type=self.cluster_type,
            #     conditions=self.all_conditions,
            #     path_to_save=path_to_save,
            #     angles=angles,
            #     delta_between_conditions=self.mangituide_deltas,
            # )
            # if analysis_name == 'PCA':
            #     logger.info("Running PCA model")
            #     run_pca_kmeans_plot(path_to_save, Preprocess_DimOBJ.x, Preprocess_DimOBJ.labels)
            #     logger.success("PCA analysis complete")

            # if analysis_name == 'UMAP':
            #     angles = identify_angles(self.session)
            #     # Run UMAP hen HDBSCAN and return the cluster ids to the neuron ids
            #     cluster_ids = run_umap_then_hdbscan(Preprocess_DimOBJ.x, Preprocess_DimOBJ.labels, save_path=path_to_save)
            #     # TODO - Compute angle similarity for each hsbscnae cluster and then assign the cluster with the highest similarity to the hdir cluster

        # ----------------------------- Persistent Homology ----------------------------------
        # if Settings.persistent_homology:
        #     persistent_homology(self.frame_by_cluster_matrix, self.video_df)

        if analysis_name == 'classify_cells':
            """Call cell type specific classification functions

            NOTE: Work in progress"""
            hdir_cell_ids = classify_hdir(session=self.session, cluster_type=self.cluster_type)
            logger.debug(f"The hdir cell ids are: {hdir_cell_ids}")

            logger.warning(f"hsa classification code for hsa is commented out and needs to be tested")
            # hsa_cell_ids = classify_hsa(
            #     session=self.session,
            #     cluster_type=self.cluster_type,
            #     hdir_cells=hdir_cell_ids,
            # )
            # logger.debug(f"The hsa cell ids are: {hsa_cell_ids}")

        # ------------------------------ Compute LSTM --------------------------------

        if analysis_name == 'LSTM':
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
        if analysis_name == 'sklearn':
            logger.info("Running Sklearn decoders")
            sklearn_main(self.session, self.video_df, self.frame_by_cluster_matrix, cluster_labels=self.cluster_Ids)

        logger.success("All models complete")

