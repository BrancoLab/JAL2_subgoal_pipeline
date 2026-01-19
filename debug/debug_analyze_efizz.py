from loguru import logger

from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from settings.settings_analyze_efizz import Settings_ae

analysis_name = 'EscapePattern'
# model_name options:
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
    # also include variable: '<var> in <context>' or 'residual: <var1> in <context1> - <var2> in <context2>'


def analyze_efizz(analysis_name=None, variable=None):
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
            aefizz.execute_models(analysis_name, variable)

    logger.success("Efizz analysis pipeline complete")

# analyze_efizz(analysis_name='EscapePattern', 
#               variable='escape in homing&escape')
analyze_efizz(analysis_name='EscapePattern', 
              variable='residual: escape in homing&escape - bird_dist_shelter in explore')

