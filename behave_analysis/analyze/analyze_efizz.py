import os
import dill as pickle
import time

from loguru import logger
import polars as pl

from settings.settings_analyze_efizz import Settings_ae as Settings
from behave_analysis.analyze.TunED.model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
# from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh, compute_single_cluster_tuning
from behave_analysis.analyze.filtering_data.filtering_functions import extract_all_or_custom_conditions, identify_angles
from behave_analysis.analyze.classification.head_direction import classify_hdir
from behave_analysis.analyze.classification.head_shelter import classify_hsa
from behave_analysis.analyze.PCA.preprocessing_pca import PreprocessPca
from behave_analysis.analyze.PCA.visulisation_pca import run_pca_kmeans_plot
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.visualize.visualize_utils import open_postprocess_object

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file.
    The purpose of this class is to make it easy to run all of the models on the same data without having to run
    the preprocessing each time. Any processing of the data should be done outside of this module.
    """

    def __init__(self, session):
        start_time = time.time()
        logger.info("Initializing AnalyzeEfizz")
        self.session = session
        self.dir = make_directory(os.path.join(session.base_path, session.processed_path,"models"))
        self.show_plots = Settings.show_plots
        self.settings = Settings
        self.all_conditions = extract_all_or_custom_conditions(Settings, session)
        self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path) + "\\"
            "full_video_dataframe.csv"
        )

        # For each cluster type in settings e.g synthetic, syntheticHdir, good, mua
        for c_type in Settings.cluster_type:
            self.cluster_type = c_type
            postprocessObject = open_postprocess_object(self.session, self.cluster_type)
            self.video_spike_count_df = postprocessObject.video_spike_count_df
            self.frame_by_cluster_matrix = postprocessObject.frame_by_cluster_matrix
            self.cluster_Ids = postprocessObject.video_spike_count_df["spike_clusters"].unique().to_numpy()

    def execute_models(self):
        logger.info("Executing models")

        # ------------------------------ Compute PCA ----------------------------------
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
            )
            run_pca_kmeans_plot(pca_path, pca.x, pca.labels)
            logger.success("PCA analysis complete")
            
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
                session = self.session,
                cluster_type=self.cluster_type,
                conditions=self.all_conditions,
            )
            logger.success("TunED analysis complete")

        # ------------------------------ Compute LSTM --------------------------------
        # TODO: Finish LSTM model

        # Run LSTM
        #         if 0:
        #             X, y = bin_polars_dataframes(spike_data = pl.read_csv(self.spike_data_frame), video_data = self.data_df)
        #             X_valid, y_valid, X_train, y_train, y_test = preprocess_data_and_set_up(neural_data = X, y = y)
        #             main(X_valid, y_valid, X_train, y_train, y_test)

        # ------------------------------ Compute LDA --------------------------------
        if len(Settings.run_LDA) > 0:
            if np.logical_or(Settings.run_LDA == 'all', 
                             np.logical_and(type(Settings.run_LDA) is list, Settings.run_LDA[0] == 'all')):
                angles = identify_angles(self.session)
                angles.append('randP')
            else: angles = Settings.run_LDA

            for o in self.all_conditions:
                self.condition = o
                logger.info(f"Run LDA on {self.cluster_type} data with condition: {self.condition}")
                run_LDA_model(self, Settings, angles)
            logger.success('LDA analysis complete')

        # ----------------- Compute Rayleigh and polar plots -------------------------
        if Settings.run_rayleigh:
            if not Settings.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                base_path = os.path.join(self.dir, 
                                         'Rayleigh', 
                                         self.cluster_type)
                compute_all_clusters_rayleigh(self, 
                                              Settings, 
                                              all_angles, 
                                              self.all_conditions, 
                                              base_path)
            else:
                logger.info(
                    f"Making single cluster polar plots on {self.cluster_type} data"
                )
                compute_single_cluster_tuning(self, Settings)

        logger.success("All models complete")

    # Had to comment out because it can't handle the Nans from the rayleigh data
    
    # def classify_cells(self):
    #     """A function to call cell type specific classification functions

    #     TODO: Work in progress"""
    #     hdir_cell_ids = classify_hdir(
    #         session=self.session, cluster_type=self.cluster_type
    #     )
    #     print("Cell ids we think are hdir", hdir_cell_ids)

    #     hsa_cell_ids = classify_hsa(
    #         session=self.session,
    #         cluster_type=self.cluster_type,
    #         hdir_cells=hdir_cell_ids,
    #     )
    #     print("Cell ids we think are hsa", hsa_cell_ids)
