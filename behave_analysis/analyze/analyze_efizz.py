# Custom libaries

from behave_analysis.analyze.TunED.tunED_model import TunEdModel
# from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
from settings.settings_analyze_efizz import Settings_ae
from behave_analysis.analyze.linshit import LinearShift
from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh, compute_single_cluster_tuning
from behave_analysis.analyze.filtering_data.filtering_functions  import identify_conditions, identify_angles
from behave_analysis.utils.creating_directories import make_directory


# OS Lib

from loguru import logger
import polars as pl
import os
import numpy as np
import dill as pickle

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file. 
    The purpose of this class is to make it easy to run all of the models on the same data without having to run 
    the preprocessing each time. Any processing of the data should be done outside of this module. 
    """
    
    def __init__(self, session):
        logger.info('Initializing AnalyzeEfizz')
        self.session = session
        self.dir = os.path.join(session.base_path, session.processed_path) + "\\" + 'models' 
        self.show_plots = Settings_ae.show_plots
        self.settings = Settings_ae
        self.all_conditions = self.extract_all_or_custom_conditions(session)
        self.video_df = pl.read_csv(os.path.join(
            self.session.base_path, 
            self.session.processed_path) + '\\' 'full_video_dataframe.csv'
                                    )
        make_directory(self.dir)
        
        # For each cluster type in settings e.g synthetic, syntheticHdir, good, mua
        for cType in Settings_ae.cluster_type:
            self.cluster_type = cType
            try: #Load in postprocess object
                fileObj = open(os.path.join(self.session.base_path, 
                                            self.session.processed_path) + "\\" + "postprocessclass" + "_" + str(self.cluster_type), 
                               'rb')
                self.postprocessObject = pickle.load(fileObj)
                fileObj.close()
            except FileNotFoundError:
                logger.error(f"Data not found for session: {self.session.name}")
                raise FileNotFoundError
            
            self.execute_models()

    def extract_all_or_custom_conditions(self, session):
        """Identify all conditions to analyze or use custom conditions from settings file"""
        if len(Settings_ae.condition) == 0:
            conditions = identify_conditions(session)
        else:
            conditions = Settings_ae.condition
        return conditions

    def execute_models(self):
        logger.info('Executing models')
        
# ------------------------------ Compute TUNED --------------------------------
        if Settings_ae.run_tunED:
            logger.info('Running TunED model')
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
            model_path = os.path.join(self.dir, 'tunED')
            TunEdModel(post_process_object = self.postprocessObject, 
                       analyze_efizz_settings =  Settings_ae, 
                       save_dir = model_path, 
                       apply_linear_shift = Settings_ae.linear_shift,
                       cluster_type = self.cluster_type,
                       conditions = self.all_conditions)
            logger.success('TunED analysis complete')
        
        # Run LSTM    
#         if 0:
#             X, y = bin_polars_dataframes(spike_data = pl.read_csv(self.spike_data_frame), video_data = self.data_df)
#             X_valid, y_valid, X_train, y_train, y_test = preprocess_data_and_set_up(neural_data = X, y = y)
#             main(X_valid, y_valid, X_train, y_train, y_test)
        
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
        if Settings_ae.run_rayleigh:
            if not Settings_ae.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                all_angles = identify_angles(self.session)
                if len(Settings_ae.condition) > 0: 
                    all_conditions = Settings_ae.condition
                else: 
                    all_conditions = identify_conditions(self.session)
                base_path = os.path.join(self.dir, 
                                         'Rayleigh', 
                                         self.cluster_type)
                compute_all_clusters_rayleigh(self, 
                                              Settings_ae, 
                                              all_angles, 
                                              all_conditions, 
                                              base_path)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self, Settings_ae)
        
        logger.success('All models complete')
            