from loguru import logger

from settings.settings_process import settings_process as settings_p
from settings.settings_track import settings_track as settings_t
from settings.settings_visualize import Settings_visualize as settings_v
from settings.settings_homings import settings_homings as settings_h
from settings.settings_analyze_efizz import Settings_ae
from settings.settings_analyze_behave import settings_analyze_behave as settings_a
from behave_analysis.process.process import Process
from behave_analysis.track.track import Track
from behave_analysis.homings.homings import get_Homings
from behave_analysis.homings.threshold_crossings import get_Threshold_crossings
from behave_analysis.visualize.visualize_efizz import Visualize_efizz
from behave_analysis.visualize.visualize_behave import Visualize_behave
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from behave_analysis.analyze.AnalyzeBehave import AnalyzeBehave
from behave_analysis.postprocess.pp_main import Postprocessor
from databank import experiments_objects


def process():
    """
    A function that collects sessions from the databank, puts the sessions through a processing
    pipeline and then saves the sessions to a metadata file. This metadata file is then loaded and used
    by subsequent track, homing, visualize and analyze functions.
    Returns: Nothing, data is saved to a metadata file.
    """
    logger.info("Processing started")
    assert len(experiments_objects) != 0, "Session list should not be empty"
    for session_ID in experiments_objects:
        logger.info("Loaded a session with the following details: {}".format(session_ID))
        processObject = Process(session_ID)
        processObject.create_session(settings_p)
    logger.success("Processing complete")
 
def track():
    """
    A function that collects sessions from the databank, puts the sessions through a tracking
    in Deep lab cut.
    """
    logger.info("The tracking pipeline has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_ID))
        Track(settings_t, session)
    logger.success("Tracking complete")

def homings():
    # TODO: Update to new databank
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        get_Homings(settings_h, session)
        get_Threshold_crossings(settings_h, session)
        
def postprocess():
    """ 
    A function that outputs and saves a postprocessed object as a pickle file in the processed data folder.
    """
    logger.info("The post processing of the data has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_ID))
        Postprocessor(session)
    logger.success("The post processing of the data has finished and the postprocessed object has been saved to a pickle file")

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
        Visualize_behave(session).plot_behavioral_stats()
        
        # ------ BEHAVIORAL VISUALIZATION ------
        if settings_v.escape_trials:
            Visualize_behave(session).make_movies(stim_type="audio")
            Visualize_behave(session).escape_plotting(stim_type="audio")
        if settings_v.homing_trials:
            Visualize_behave(session).make_movies(stim_type="homing")

        # ------ EFIZZ VISUALIZATION ------
        if settings_v.efizz:
            Visualize_efizz(session).run_tuning_functions()
            if settings_v.stim_type != "None":
                Visualize_efizz(session).run_stim_resp_plotting()
    logger.success("Visualisation complete")

def analyze_efizz(analysis_name=None):
    """A function that runs the specified analysis on the data.
        It is designed to be run last and for the whole dataset.
        # analysis_name options:
        # 'LDA' - Linear Discriminant Analysis
        # 'single_trial' - Single Trial Analysis
        # 'rayleigh' - Rayleigh Analysis
        # 'tunED' - TunEd Analysis
        # 'PCA' or 'UMAP' - Dimentionality Reduction
        # 'classify_cells' - Cell Type Classification (currently only works for HD cells)
        # 'sklearn' - Sklearn Decoders"""

    logger.info("Initiating {} analysis pipeline".format(analysis_name))

    if analysis_name is None:
        logger.warning("No analysis name provided - skipping efizz analyses")
        return
    
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        for c_type in Settings_ae.cluster_type:
            AnalyzeEfizz(session, c_type).execute_models(analysis_name)

    logger.success("Efizz analysis pipeline complete")

def analyze_behave():
    """A function that calls all the analysis modules and is designed to be run last and for the whole dataset."""
    
    logger.info("The behaviour analysis pipeline has started")
    
    if settings_a.stim_type == "None":
        logger.warning("No stim type defined in settings - skipping behavioral analyses")
        return
    
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        AnalyzeBehave(session).behaviour_analyses()

    logger.success("Behaviour analysis pipeline complete")
            