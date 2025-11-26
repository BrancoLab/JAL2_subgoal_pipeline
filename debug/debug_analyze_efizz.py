from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from settings.settings_analyze_efizz import Settings_ae

analysis_name = 'EscapePattern'
# model_name options:
# 'LDA' - Linear Discriminant Analysis
# 'single_trial' - Single Trial Analysis
# 'rayleigh' - Rayleigh Analysis
# 'tunED' - TunEd Analysis
# 'PCA' or 'UMAP' - Dimentionality Reduction
# 'classify_cells' - Cell Type Classification (currently only works for HD cells)
# 'sklearn' - Sklearn Decoders
# 'EscapePattern' - Escape Pattern Tuning Analysis


def analyze_efizz(analysis_name=None):
    """A function that runs the specified analysis on the data.
      It is designed to be run last and for the whole dataset."""

    logger.info("Initiating {} analysis pipeline".format(analysis_name))

    if analysis_name is None:
        logger.warning("No analysis name provided - skipping efizz analyses")
        return
    
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        for c_type in Settings_ae.cluster_type:
            aefizz = AnalyzeEfizz(session, c_type)
            aefizz.load_data(analysis_name)
            aefizz.execute_models(analysis_name)

    logger.success("Efizz analysis pipeline complete")


analyze_efizz('EscapePattern')

