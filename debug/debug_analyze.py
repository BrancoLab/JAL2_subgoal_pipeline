from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from settings.settings_analyze import settings_analyze as settings_a
from settings.settings_analyze_efizz import Settings_ae


def analyze():
    """A function that calls all the analysis modules and is designed to be run last and for the whole dataset."""
    logger.info("The analysis pipeline has started")
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        if Settings_ae.stim_type != "None":
            AnalyzeBehave(session).behaviour_analyses()

        if settings_a.efizz:
            for c_type in Settings_ae.cluster_type:
                
                AnalyzeEfizz(session, c_type).execute_models()
                if Settings_ae.classify_cells:
                    AnalyzeEfizz(session, c_type).classify_cells()
                
                
    logger.success("Analysis pipeline complete")


analyze()
