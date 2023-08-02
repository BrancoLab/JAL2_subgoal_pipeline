
from behave_analysis.analyze.TunED.tunED_model import TunEdModel
from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
# from behave_analysis.analyze.ConSink.Consink_model import Consink
from settings.settings_analyze_efizz import Settings_analyze_efizz
from behave_analysis.analyze.linshit import LinearShift
from behave_analysis.analyze.decoders.LSTM.LSTM_model import preprocess_data_and_set_up, main, bin_polars_dataframes

# OS Lib
from loguru import logger
import polars as pl
import os

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file. The purpose
    of this class is to make it easy to run all of the models on the same data without having to run the preprocessing each time.
    Any processing of the data should be done outside of this module. 
    """
    def __init__(self, session):
        logger.info('Initializing AnalyzeEfizz')
        self.dir = session.processed_path + "\\" + 'models' 
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.show_plots = Settings_analyze_efizz.show_plots
        cluster_type = Settings_analyze_efizz.cluster_type
        object_present = Settings_analyze_efizz.object_present
        for c in cluster_type:
            for o in object_present:
                self.object_present = o
                self.cluster_type = c
                logger.info(f"Running models on cluster category: {self.cluster_type}")
                self.processed_file_directory = session.processed_path + '\\' + str(self.cluster_type) + '_large_dataframe.csv'
                self.spike_data_frame = session.processed_path + '\\' + "synthetic_efizz_data.csv" # Per spike data not binned - Need to update to be dynamic NOTE
                self.video_data_frame = pl.read_csv(session.processed_path + '\\' + "spike_count_by_frame_and_syntheticcluster.csv") # video frame NOTE update
                if os.path.isfile(self.processed_file_directory):
                    self.data_df = pl.read_csv(self.processed_file_directory)
                else:
                    raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
                self.execute_models()

    def execute_models(self):
        logger.info('Executing models')
        
        if Settings_analyze_efizz.run_tunED:
            if not os.path.isdir(self.dir + "\\" + "tunED"):
                os.mkdir(self.dir + "\\" + "tunED")
                
            model_path = os.path.join(self.dir, 'tunED')
            logger.info('Running TunED')
            
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

        # if Settings_analyze_efizz.run_consink:
        #     logger.info('Running Consink')
        #     Consink(self)
        #     logger.success('Consink analysis complete')
        
        # if Settings_analyze_efizz.run_pcaGLM:
            # logger.info('Running pcaGLM')
            # pcaGLM(self.large_data_file)
            # logger.success('pcaGLM analysis complete')
            
        if len(Settings_analyze_efizz.run_LDA) > 0:
            logger.info(f"Run LDA on {self.cluster_type} data with object_present: {self.object_present}")
            run_LDA_model(self,Settings_analyze_efizz)
            logger.success('LDA analysis complete')
        
        logger.success('All models complete')
            