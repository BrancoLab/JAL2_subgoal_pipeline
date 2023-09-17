
from behave_analysis.analyze.TunED.tunED_model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
from settings.settings_analyze_efizz import Settings_analyze_efizz
from behave_analysis.analyze.linshit import LinearShift
from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes
from behave_analysis.analyze.Rayleigh.computeRayleigh import compute_all_clusters_rayleigh, compute_single_cluster_tuning
from behave_analysis.analyze.filtering_data.filtering_functions  import identify_conditions

# OS Lib
from loguru import logger
import polars as pl
import os
import numpy as np

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file. The purpose
    of this class is to make it easy to run all of the models on the same data without having to run the preprocessing each time.
    Any processing of the data should be done outside of this module. 
    """
    def __init__(self, session):
        logger.info('Initializing AnalyzeEfizz')
        self.dir = session.processed_path + "\\" + 'models' 
        self.session = session
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.show_plots = Settings_analyze_efizz.show_plots
        self.settings = Settings_analyze_efizz
        # cluster_type = Settings_analyze_efizz.cluster_type
        # check which conditions the user wants us to use
        if len(Settings_analyze_efizz.condition) == 0:
            self.all_conditions = identify_conditions(session)
        else:
            self.all_conditions = Settings_analyze_efizz.condition
        for c in Settings_analyze_efizz.cluster_type:
            self.cluster_type = c
            self.execute_models()

    def execute_models(self):
        logger.info('Executing models')
        
        if Settings_analyze_efizz.run_tunED:
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
                
            model_path = os.path.join(self.dir, 'tunED')
            logger.info('Running TunED')

            # load data
            # self.spike_data_frame = self.session.processed_path + '\\' + "synthetic_efizz_data.csv" # Per spike data not binned - Need to update to be dynamic NOTE
            # self.video_data_frame = pl.read_csv(self.session.processed_path + '\\' + "spike_count_by_frame_and_syntheticcluster.csv") # video frame NOTE update
            self.processed_file_directory = self.session.processed_path + '\\' + str(self.cluster_type) + '_large_dataframe.csv'
            if os.path.isfile(self.processed_file_directory):
                self.data_df = pl.read_csv(self.processed_file_directory)
            else:
                raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
            
            TunEdModel(self, 
                       analyze_efizz_settings =  Settings_analyze_efizz, 
                       save_location = model_path, 
                       apply_linear_shift = False,
                       save_plots = False)
              
            logger.success('TunED analysis complete')
        
        # Run LSTM    
#         if 0:
#             X, y = bin_polars_dataframes(spike_data = pl.read_csv(self.spike_data_frame), video_data = self.data_df)
#             X_valid, y_valid, X_train, y_train, y_test = preprocess_data_and_set_up(neural_data = X, y = y)
#             main(X_valid, y_valid, X_train, y_train, y_test)
            
        if len(Settings_analyze_efizz.run_LDA) > 0:
            for o in self.all_conditions:
                self.condition = o
                logger.info(f"Run LDA on {self.cluster_type} data with condition: {self.condition}")
                # load data
                self.video_df = pl.read_csv(self.session.processed_path + '\\' 'full_video_dataframe.csv')
                self.processed_file_directory = self.session.processed_path + '\\' 'frame_by_' + str(self.cluster_type) + '_cluster_matrix.npy'
                self.firing_matrix = np.load(self.processed_file_directory)
                run_LDA_model(self,Settings_analyze_efizz)
            logger.success('LDA analysis complete')

        if Settings_analyze_efizz.run_rayleigh:
            # load data
            self.processed_file_directory = self.session.processed_path + '\\' + str(self.cluster_type) + '_large_dataframe.csv'
            if os.path.isfile(self.processed_file_directory):
                self.data_df = pl.read_csv(self.processed_file_directory)
            else:
                raise FileNotFoundError("Data file doesn't exsist, have you generated it?")
            if not Settings_analyze_efizz.single_cluster_plots:
                logger.info(f"Compute Rayleigh on {self.cluster_type} data")
                compute_all_clusters_rayleigh(self,Settings_analyze_efizz)
            else:
                logger.info(f"Making single cluster polar plots on {self.cluster_type} data")
                compute_single_cluster_tuning(self,Settings_analyze_efizz)
        
        logger.success('All models complete')
            