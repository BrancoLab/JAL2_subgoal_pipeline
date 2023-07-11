from behave_analysis.track.track import Track
from behave_analysis.process.process import Process
from loguru import logger
from databank import experiments_objects
from settings.settings_track import settings_track as settings_t

def track():
    """
    A function that collects sessions from the databank, puts the sessions through a tracking
    in Deep lab cut.
    """
    logger.info("The tracking pipeline has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        Track(settings_t, session)
    logger.success("Tracking complete")
    
track()