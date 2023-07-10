from behave_analysis.analyze.TunED.tunED_model import tunED_model_main
from behave_analysis.analyze.LDA.LDAmodel import linear_discriminant_analysis
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
    
    # TODO make flag for synth vs production better 
    def __init__(self, session):
        logger.info('Initializing AnalyzeEfizz')
        self.dir = session.processed_path + '/models'
        self.cluster_type = Settings_analyze_efizz.cluster_type
        self.show_plots = Settings_analyze_efizz.show_plots
        self.processed_file_directory = session.processed_path + '/' + str(Settings_analyze_efizz.cluster_type) + '_large_dataframe.csv'
        if os.path.isfile(self.processed_file_directory):
            self.data_df = pl.read_csv(self.processed_file_directory)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
        self.execute_models()

    def execute_models(self):
        logger.info('Executing models')
        
        if Settings_analyze_efizz.run_tunED:
            logger.info('Running TunED')
            tunED_model_main(self.data_df, file_save_location = self.dir  / 'tuned')
            logger.success('TunED analysis complete')
            
        if Settings_analyze_efizz.run_consink:
            logger.info('Running Consink')
            Consink(self)
            logger.success('Consink analysis complete')
        
        # if Settings_analyze_efizz.run_pcaGLM:
            # logger.info('Running pcaGLM')
            # pcaGLM(self.large_data_file)
            # logger.success('pcaGLM analysis complete')
            
        if len(Settings_analyze_efizz.run_LDA) > 0:
            for variable in Settings_analyze_efizz.run_LDA:
                logger.info(f"Running LDA on {variable}")
                linear_discriminant_analysis(self, variable,Settings_analyze_efizz.object_present)
                logger.success('LDA analysis complete')
        
        logger.success('All models complete')
            