import os
import dill as pickle

from loguru import logger
import polars as pl
import numpy as np

from behave_analysis.analyze.TunED.model import TunEdModel

# from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
from settings.settings_analyze_efizz import Settings_ae as Settings

from behave_analysis.analyze.decoders.LSTM.pytorch_LSTM import main

# from behave_analysis.analyze.decoders.LSTM.LSTM_model import main_new
from behave_analysis.analyze.Rayleigh.computeRayleigh import (
    compute_all_clusters_rayleigh,
    compute_single_cluster_tuning,
)
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, identify_angles
from behave_analysis.analyze.classification.head_direction import classify_hdir
from behave_analysis.utils.creating_directories import make_directory


class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file.
    The purpose of this class is to make it easy to run all of the models on the same data without having to run
    the preprocessing each time. Any processing of the data should be done outside of this module.
    """

    def __init__(self, session):
        logger.info("Initializing AnalyzeEfizz")
        self.session = session
        self.dir = os.path.join(session.base_path, session.processed_path) + "\\" + "models"
        self.show_plots = Settings.show_plots
        self.settings = Settings
        self.all_conditions = self.extract_all_or_custom_conditions(session)
        self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv"
        )
        make_directory(self.dir)

        # For each cluster type in settings e.g synthetic, syntheticHdir, good, mua
        for c_type in Settings.cluster_type:
            self.cluster_type = c_type
            try:  # Load in postprocess object
                fileObj = open(
                    os.path.join(self.session.base_path, self.session.processed_path)
                    + "\\"
                    + "postprocessclass"
                    + "_"
                    + str(self.cluster_type),
                    "rb",
                )
                self.postprocessObject = pickle.load(fileObj)
                fileObj.close()
            except FileNotFoundError:
                logger.error(f"Data not found for session: {self.session.name}")
                raise FileNotFoundError

            self.execute_models()
            self.classify_cells()

    def extract_all_or_custom_conditions(self, session):
        """Identify all conditions to analyze or use custom conditions from settings file"""
        if len(Settings.condition) == 0:
            conditions = identify_conditions(session)
        else:
            conditions = Settings.condition
        return conditions

    def execute_models(self):
        logger.info("Executing models")

        # ------------------------------ Compute TUNED --------------------------------
        if Settings.run_tunED:
            logger.info("Running TunED model")
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
            model_path = os.path.join(self.dir, "tunED")
            TunEdModel(
                post_process_object=self.postprocessObject,
                analyze_efizz_settings=Settings,
                save_dir=model_path,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
            )
            logger.success("TunED analysis complete")

        # ------------------------------ Compute LSTM --------------------------------
        # TODO: Finish LSTM model
        
        main(frame_by_cluster_matrix = self.postprocessObject.frame_by_cluster_matrix, 
             Y = np.asarray(self.video_df["hdir"]).reshape(len(self.video_df["hdir"]), 1))

        # X, y = bin_polars_dataframes(spike_data = pl.read_csv(self.spike_data_frame), video_data = self.data_df)
        # X_valid, y_valid, X_train, y_train, y_test = preprocess_data_and_set_up(neural_data = X, y = y)
        # main(X_valid, y_valid, X_train, y_train, y_test)

        # ------------------------------ Compute LDA --------------------------------
        # if len(Settings_analyze_efizz.run_LDA) > 0:
        #     if Settings_analyze_efizz.run_LDA == 'all':
        #         angles = identify_angles(self.session)
        #         angles.append('randP')
        #     else: angles = Settings_analyze_efizz.run_LDA

        #     for o in self.all_conditions:
        #         self.condition = o
        #         logger.info(f"Run LDA on {self.cluster_type} data with condition: {self.condition}")
        #         run_LDA_model(self,Settings_analyze_efizz, angles)
        #     logger.success('LDA analysis complete')

        # ----------------- Compute Rayleigh and polar plots -------------------------
        if Settings.run_rayleigh:
            if not Settings.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                if len(Settings.condition) > 0:
                    all_conditions = Settings.condition
                else:
                    all_conditions = identify_conditions(self.session)
                base_path = os.path.join(self.dir, "Rayleigh", self.cluster_type)
                compute_all_clusters_rayleigh(self, Settings, all_angles, all_conditions, base_path)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self, Settings)

        logger.success("All models complete")
        
    def classify_cells(self):
        """A function to call cell type specific classification functions
        
        TODO: Work in progress"""
        hdir_cell_ids = classify_hdir(session = self.session, cluster_type = self.cluster_type)
        print("Cell ids we think are hdir", hdir_cell_ids)
