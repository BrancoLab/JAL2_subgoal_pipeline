from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from settings.settings_analyze import Settings_analyze as settings_a


def analyze():
    """A function that calls all the analysis modules and is designed to be run last and for the whole dataset."""
    logger.info("The analysis pipeline has started")
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))
        if settings_a.stim_type != "None":
            AnalyzeBehave(session).behaviour_analyses()
        # if settings_a.efizz:
        #     AnalyzeEfizz(session).execute_models()
            # AnalyzeEfizz(session).classify_cells()
    logger.success("Analysis pipeline complete")


analyze()
