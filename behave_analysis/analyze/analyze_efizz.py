import os
import time

import numpy as np
from loguru import logger
import polars as pl
import pickle
import matplotlib.pyplot as plt

# from behave_analysis.analyze.decoders.pytorch.lstm_main import main
from behave_analysis.analyze.TunED.model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model, across_conditions_LDA_map
from settings.settings_analyze_efizz import Settings_ae as Settings
# from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh, compute_single_cluster_tuning
from behave_analysis.analyze.filtering_data.filtering_functions import extract_all_or_custom_conditions, identify_angles
from behave_analysis.analyze.classification.head_direction import classify_hdir
from behave_analysis.analyze.classification.head_shelter import classify_hsa
from behave_analysis.analyze.PCA.preprocessing_pca import PreprocessPca
from behave_analysis.analyze.PCA.visulisation_pca import run_pca_kmeans_plot
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.visualize.visualize_utils import open_postprocess_object, open_tracking_data
from behave_analysis.analyze.regression_decoders.sklearn_decoders.sk_models import rf_model, svr_model, gbr_model, elastic_net_model
from behave_analysis.analyze.regression_decoders.sklearn_decoders.input import gen_random_pred_array, split_data
from behave_analysis.analyze.regression_decoders.sklearn_decoders.sklearn_main import sklearn_main
from behave_analysis.analyze.Rayleigh.analyze_rayleighs import plot_rayleigh_deltas
from behave_analysis.visualize.visualize_utils import open_postprocess_object


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
        assert c_type in ["synthetic", "syntheticHdir", "all", "good", "mua", "noise"], "Cluster type not recognised"
        assert os.path.isfile(
            os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + c_type + "_cluster_matrix.npy"
        ), "Cluster matrix file not found"
        self.frame_by_cluster_matrix = np.load(
                os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "frame_by_" + c_type + "_cluster_matrix.npy"
            )
        self.tracking_data = open_tracking_data(self.session)
        # self.cluster_Ids = np.load(str(os.path.join(self.session.base_path,self.session.processed_path) + "/" + self.cluster_type + "_cluster_Ids.npy"))

        logger.info("Loading giant post processing object this will take for ever")
        # postprocessObject = open_postprocess_object(self.session, self.cluster_type)
            # self.video_spike_count_df = postprocessObject.video_spike_count_df
            # self.frame_by_cluster_matrix = postprocessObject.frame_by_cluster_matrix
        # self.cluster_Ids = postprocessObject.clu_label["spike_clusters"].unique().to_numpy()
            # self.tracking_data = postprocessObject.tracking_data

    def execute_models(self):
        logger.info("Executing models")
        
        
        # ----------------- Compute Rayleigh, polar plots and delta hists ------------
        if Settings.run_rayleigh:
            if not Settings.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                base_path = os.path.join(self.dir, "Rayleigh", self.cluster_type)
                compute_all_clusters_rayleigh(self, Settings, all_angles, self.all_conditions, base_path)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self, Settings)
                
        self.mangituide_deltas = plot_rayleigh_deltas(self.session, self.cluster_type) # Analyze rayleigh deltas
        
    
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
        # TODO: Finish LSTM model

        # if Settings.run_LSTM:
        #     logger.info("Running LSTM model")

        #     # Use buzacki data instead of ours ------------------------------
        #     # save spike_rate_cell and angles to a pickle file
        #     file_location = r"E:\\efizz\\JAL004\\004_flipppuf19sept_2023_09_19T14_10_56\\processed_data\\buzacki_data"
        #     with open(file_location + "\\" "spike_rate_cell.p", "rb") as f:
        #         spike_rate_cell = pickle.load(f)

        #     with open(file_location + "\\" + "angles.p", "rb") as f:
        #         angles = pickle.load(f)

        #     y_reshaped = np.asarray(angles).reshape(len(angles), 1)
        #     y_adjusted = np.nan_to_num(y_reshaped) - np.pi
        #     x = spike_rate_cell
        #     # main(x, y_adjusted)

        # ------------------------------ Sklearn decoder models --------------------------------
        if Settings.run_sklearn_decoders:
            sklearn_main(self.session, self.video_df, self.frame_by_cluster_matrix, cluster_labels=self.cluster_Ids)

        # ------------------------------ Compute LDA --------------------------------
        if len(Settings.run_LDA) > 0:
            if np.logical_or(Settings.run_LDA == "all", np.logical_and(type(Settings.run_LDA) is list, Settings.run_LDA[0] == "all")):
                angles = identify_angles(self.session)
                angles.append("randP")
            else:
                angles = Settings.run_LDA

            for o in self.all_conditions:
                self.condition = o
                logger.info(f"Run LDA on {self.cluster_type} data with condition: {self.condition}")
                run_LDA_model(self, Settings, angles)
            across_conditions_LDA_map(self, Settings)
            logger.success('LDA analysis complete')

        # ----------------------------- Compute PCA ----------------------------------
        if Settings.run_pca_model:
            logger.info("Running PCA model")
            pca_path = os.path.join(self.dir, "PCA")
            angles = identify_angles(self.session)
            make_directory(pca_path)
            pca = PreprocessPca(
                session=self.session,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
                path_to_save=pca_path,
                angles=angles,
                delta_between_conditions = self.mangituide_deltas
            )
            run_pca_kmeans_plot(pca_path, pca.x, pca.labels)
            logger.success("PCA analysis complete")

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
