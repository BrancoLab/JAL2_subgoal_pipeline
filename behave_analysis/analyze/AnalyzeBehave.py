from loguru import logger
import polars as pl
import os

# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from settings.settings_analyze import settings_analyze as settings
from behave_analysis.utils.creating_directories import make_directory

class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """
    def __init__(self,session):
        logger.info('Initializing AnalyzeBehave')
        self.dir = make_directory(os.path.join(session.base_path,session.processed_path) + "\\" + 'analyze_behave')
        self.session = session
        self.settings = settings
        open_tracking_data(self)
        """Load in video df"""
        video_df = os.path.join(session.base_path,session.processed_path) + "\\" + "full_video_dataframe.csv"
        if os.path.isfile(video_df):
            self.video_df = pl.read_csv(video_df)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
 
    def behaviour_analyses(self):
        logger.info(f"Making plots of spatial effciency in escape")
        spatial_efficiency(session = self.session, 
                           settings = settings, 
                           video_df = self.video_df, 
                           tracking_data = self.tracking_data, 
                           save_dir = self.dir)