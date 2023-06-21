from behave_analysis.analyze.TunED.main import tunED_main
from settings.settings_analyze_efizz import Settings_analyze_efizz

# OS Lib
from loguru import logger
import polars as pl

class AnalyzeEfizz:
    """
    A class that loads already processed efizz data and then runs all of the models on it set in the settings file. The purpose
    of this class is to make it easy to run all of the models on the same data without having to run the preprocessing each time.
    Any processing of the data should be done outside of this module. 
    """
    def __init__(self, session_to_analyze):
        logger.info('Initializing AnalyzeEfizz')
        self.processed_file_directory = session_to_analyze.file_path / 'processed_data'
        # self.large_data_file = pl.read_csv(self.processed_file_directory / "_Production_large_dataframe.csv")
        self.large_data_file = pl.read_csv(self.processed_file_directory / "Test_large_dataframe.csv")
        self.execute_models()

    def execute_models(self):
        logger.info('Executing models')
        
        if Settings_analyze_efizz.run_tunED:
            logger.info('Running TunED')
            tunED_main(self.large_data_file)
            logger.success('TunED analysis complete')
            
        # if Settings_analyze_efizz.run_consink:
            # logger.info('Running Consink')
            # Consink(self.large_data_file)
            # logger.success('Consink analysis complete')
        
        # if Settings_analyze_efizz.run_pcaGLM;
            # logger.info('Running pcaGLM')
            # pcaGLM(self.large_data_file)
            # logger.success('pcaGLM analysis complete')
            
        # if Settings_analyze_efizz.run_LDA;
            # logger.info('Running LDA')
            # LDA(self.large_data_file)
            # logger.success('LDA analysis complete')
        
        logger.success('All models complete')
            