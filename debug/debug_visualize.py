"""A debug script for visualizing mouse behavior and efizz data""" ""

from loguru import logger

from databank import experiments_objects
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.process.process import Process
from behave_analysis.visualize.visualize_efizz import Visualize_efizz
from behave_analysis.visualize.visualize_behave import Visualize_behave

def visualize():
    """Viusalize mouse behavior and efizz data
    
    Responsibilities:
    -- Create movies of each trial type (homing, escapes)
    -- plot some behavioral statistics
    -- plot some efizz statistics
    """

    logger.info("Visualisation started")
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))
        # Visualize_behave(session).plot_behavioral_stats()

        # ------ BEHAVIORAL VISUALIZATION ------
        if settings_v.escape_trials:
            Visualize_behave(session).make_movies(stim_type="audio")
            Visualize_behave(session).escape_plotting()
        if settings_v.homing_trials:
            Visualize_behave(session).make_movies(stim_type="homing")

        # ------ EFIZZ VISUALIZATION ------
        if settings_v.efizz:
            Visualize_efizz(session).run_tuning_functions()
            if settings_v.stim_type != "None":
                Visualize_efizz(session).run_stim_resp_plotting()

    logger.success("Visualisation pipeline step complete")


visualize()
