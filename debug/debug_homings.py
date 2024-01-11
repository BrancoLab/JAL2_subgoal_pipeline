"""Debugging homings and threshold crossings."""
from loguru import logger

from databank import experiments_objects
from settings.settings_homings import settings_homings as settings_h
from behave_analysis.process.process import Process
from behave_analysis.homings.homings import get_Homings
# from behave_analysis.homings.threshold_crossings import get_Threshold_crossings

# Define the homings function
def homings():
    """Let's check out some homings and threshold crossings."""
    logger.info("The homings pipeline has started")
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        get_Homings(settings_h, session)
        # get_Threshold_crossings(settings_h, session)
    logger.success("Homing pipeline complete")

# Now let's run it
homings()
