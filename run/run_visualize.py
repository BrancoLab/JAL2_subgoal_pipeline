
"""A debug script for visualizing mouse behavior and efizz data""" ""

from loguru import logger

from databank import experiments_objects
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.process.process import Process
from behave_analysis.visualize.visualize_efizz import Visualize_efizz
from behave_analysis.visualize.visualize_behave import Visualize_behave

VISUALIZATIONS = ["escape_plotting", 
                  "homing_plotting", 
                  "spatial_position_firing", 
                  "spatial_position_firing_hdir",
                  "single_unit_heatmaps",
                  "pop_rasters",
                  "pop_PSTH",
                  "PSTH_single_cluster",
                  "single_cluster_raster"]

def visualize(visualization_name=None):
    """Viusalize mouse behavior and efizz data
    
    Responsibilities:
    -- Create movies of each trial type (homing, escapes)
    -- plot some behavioral statistics
    -- plot some efizz statistics
    """

    logger.info("Visualisation started")
    if visualization_name not in VISUALIZATIONS:
        logger.warning("No valid visualization name provided - skipping efizz visualizations")
        return
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))
        #Visualize_behave(session).plot_behavioral_stats()

        # # ------ BEHAVIORAL VISUALIZATION ------
        if visualization_name == "escape_plotting":
            Visualize_behave(session).make_movies(stim_type="audio")
            Visualize_behave(session).escape_plotting(stim_type="audio")
        if visualization_name == "homing_plotting":
            Visualize_behave(session).make_movies(stim_type="homing")

        # ------ EFIZZ VISUALIZATION ------
        else:
            vefizz = Visualize_efizz(session)
            vefizz.load_data(visualization_name)
            vefizz.run_visualizations(visualization_name)

    logger.success("Visualisation pipeline step complete")

visualize()
