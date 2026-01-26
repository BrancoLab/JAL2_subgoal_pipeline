from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from settings.settings_analyze_behave import Settings


def analyze_behave():
    """A function that calls all the analysis modules and is designed to be run last and for the whole dataset."""
    
    logger.info("The behaviour analysis pipeline has started")
    
    if Settings.stim_type == "None":
        logger.warning("No stim type defined in settings - skipping behavioral analyses")
        return
    
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        AnalyzeBehave(session).behaviour_analyses()

    logger.success("Behaviour analysis pipeline complete")

analyze_behave()
