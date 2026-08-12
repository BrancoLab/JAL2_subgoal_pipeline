from behave_analysis.process.process import Process
from loguru import logger
from databank import experiments_objects
from settings.settings_process import settings_process

def process(experiments_objects, settings_p):
    """A function that collects sessions from the databank, puts the sessions through a processing
    pipeline and then saves the sessions to a metadata file. This metadata file is then loaded and used
    by subsequent track, homing, visualize and analyze functions.
    Returns: Nothing, data is saved to a metadata file."""
    
    assert len(experiments_objects) != 0, "Session list should not be empty"
    for session_id in experiments_objects:
        logger.info(f"Processing session: {session_id.nick_name} on {session_id.experiment_date} for experiment: {session_id.experiment_name}")
        processObject = Process(session_id)
        processObject.create_session(settings_p)
    logger.success("Processing complete")

process(experiments_objects, settings_p=settings_process)