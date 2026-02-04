from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from settings.settings_analyze_efizz import Settings_ae
import copy

# analysis_name options:
# 'LDA' - Linear Discriminant Analysis
    # variable: 'all_angles', 'all_distance','all_vectors' it will run it for all possible angles, distances, vectors
    # else:  list of angles ['hsa','hdir','h_postflipbar_a','h_preflipbar_a','h_bar_centre_a', 'randP']
    # TODO: linear shift doesn't curently work for vect or dist because of binning and other inputs neede in linear_discriminant_analysis function
# 'single_trial' - Single Trial Analysis
# 'rayleigh' - Rayleigh Analysis
# 'tunED' - TunED Analysis
# 'PCA' or 'UMAP' - Dimentionality Reduction
# 'classify_cells' - Cell Type Classification (currently only works for HD cells)
# 'sklearn' - Sklearn Decoders
# 'EscapePattern' - Escape Pattern Tuning Analysis
    # variable: e.g. '<var> in <context>' or 'residual: <var1> in <context1> - <var2> in <context2>'


def analyze_efizz(analysis_name=None, variable=None, overrides=None):
    """A function that runs the specified analysis on the data.
      It is designed to be run last and for the whole dataset.
      INPUTS:
        analysis_name: str, name of the analysis to run
        variable: str, variable to analyze (if applicable)
        overrides: dict, settings to override default settings
    """

    if analysis_name is None:
        logger.warning("No analysis name provided - skipping efizz analyses")
        return
    logger.info("Initiating {} analysis pipeline".format(analysis_name))

    settings = settings_overrides(overrides)

    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))

        aefizz = AnalyzeEfizz(session, settings)
        aefizz.load_data(analysis_name)
        aefizz.execute(analysis_name, variable)

    logger.success("Efizz analysis pipeline complete")

def settings_overrides(overrides=None):
    """Function to override default settings for efizz analysis"""
    settings = copy.deepcopy(Settings_ae)
    if overrides:
        for k, v in overrides.items():
            if hasattr(settings, k):
                setattr(settings, k, v)
            else:
                logger.warning("Unknown setting %s", k)

    return settings

analyze_efizz(analysis_name='Replay')
# analyze_efizz(analysis_name='classify_cells')
# analyze_efizz(analysis_name='EscapePattern', 
#               variable='tuned: escape in homing&escape - bird_dist_shelter in homing&escape')

