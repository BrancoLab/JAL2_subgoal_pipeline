from behave_analysis.process.process import Process
from loguru import logger
from databank import experiments_objects
from settings.settings_process import settings_process as settings_p

def process():
    """A function that collects sessions from the databank, puts the sessions through a processing
    pipeline and then saves the sessions to a metadata file. This metadata file is then loaded and used
    by subsequent track, homing, visualize and analyze functions.
    Returns: Nothing, data is saved to a metadata file."""
    
    logger.info("Processing started")
        
    assert len(experiments_objects) != 0, "Session list should not be empty"

    for session_ID in experiments_objects:
        Process(session_ID).create_session(settings_p)
        
    logger.success("Processing complete")
    
process()