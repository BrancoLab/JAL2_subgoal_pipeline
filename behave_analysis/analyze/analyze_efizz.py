from behave_analysis.analyze.TunED.tunED_model import tunED_model_main
from behave_analysis.analyze.LDA.LDAmodel import run_LDA_model
from behave_analysis.analyze.ConSink.Consink_model import Consink
from settings.settings_analyze_efizz import Settings_analyze_efizz

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
        self.dir = session.processed_path + "\\" + 'models' + "\\" + "tunED"
        self.cluster_type = Settings_analyze_efizz.cluster_type
        self.show_plots = Settings_analyze_efizz.show_plots
        self.processed_file_directory = session.processed_path + '\\' + str(Settings_analyze_efizz.cluster_type) + '_large_dataframe.csv'
        if os.path.isfile(self.processed_file_directory):
            self.data_df = pl.read_csv(self.processed_file_directory)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
        self.execute_models()

    def execute_models(self):
        logger.info('Executing models')
        
        # Params
        objectPresent = Settings_analyze_efizz.object_present
        
        if Settings_analyze_efizz.run_tunED:
            logger.info('Running TunED')
            tunED_model_main(self.data_df, objectPresent, file_save_location = self.dir)
            logger.success('TunED analysis complete')
            
        # if Settings_analyze_efizz.run_consink:
        #     logger.info('Running Consink')
        #     Consink(self)
        #     logger.success('Consink analysis complete')
        
        # if Settings_analyze_efizz.run_pcaGLM:
            # logger.info('Running pcaGLM')
            # pcaGLM(self.large_data_file)
            # logger.success('pcaGLM analysis complete')
            
        if len(Settings_analyze_efizz.run_LDA) > 0:
            run_LDA_model(self,Settings_analyze_efizz)
            logger.success('LDA analysis complete')
        
        logger.success('All models complete')
            