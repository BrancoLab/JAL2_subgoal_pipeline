from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from settings.settings_analyze_behave import settings_ab


def analyze_behave(analysis_name = None):
    """A function that calls all the analysis modules and is designed to be run last and for the whole dataset."""
    
    logger.info("The behaviour analysis pipeline has started")
    
    if settings_ab.escape_stim_type == "None":
        logger.warning("No stim type defined in settings - skipping behavioral analyses")
        return
    
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        abehave = AnalyzeBehave(session, settings_ab)
        abehave.load_data(analysis_name)
        abehave.behaviour_analyses(analysis_name)

    logger.success("Behaviour analysis pipeline complete")

analyze_behave('homings&escape')
