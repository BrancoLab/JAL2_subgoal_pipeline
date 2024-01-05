from loguru import logger
import polars as pl
import os

# Custom classes
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from settings.settings_analyze import settings_analyze as settings

class AnalyzeBehave:
    """
    A class that analyzes mouse behavior in a session
    """
    def __init__(self,session):
        logger.info('Initializing AnalyzeBehave')
        self.dir = os.path.join(session.base_path,session.processed_path) + "\\" + 'analyze_behave' 
        self.session = session
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.show_plots = settings.show_plots
        self.settings = settings
        open_tracking_data(self)
        """Load in video df"""
        video_df = os.path.join(session.base_path,session.processed_path) + "\\" + "full_video_dataframe.csv"
        if os.path.isfile(video_df):
            self.video_df = pl.read_csv(video_df)
        else:
            raise FileNotFoundError("Synthetic data path doesn't exsist, have you generated it?")
 
    def behaviour_analyses(self):
        spatial_efficiency()