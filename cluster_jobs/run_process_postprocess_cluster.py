import sys
from pathlib import Path
from loguru import logger

# Ensure repository-root imports (e.g. databank, run, settings) work when this
# script is invoked as a file path from cluster jobs.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from databank import cluster_experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.postprocess.pp_main import Postprocessor
from settings.settings_process import settings_process
from run.run_analyze_efizz import analyze_efizz

assert len(cluster_experiments_objects) != 0, "Session list should not be empty"

for session_id in cluster_experiments_objects:
	logger.info("Creating a session with the following details: {}", session_id)
	process_object = Process(session_id)
	process_object.create_session(settings_process)
logger.success("Processing complete")

for session_id in cluster_experiments_objects:
	session = Process(session_id).load_session()
	logger.info("Loaded a session with the following details: {}", session_id)
	Postprocessor(session)
logger.success("Postprocessing complete")


# analyze efizz data
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from settings.settings_analyze_efizz import Settings_ae

analysis_name='EscapePattern'
variable = ['frac_route in homing&escape', 'frac_route in to_subgoal_homing&escape', 'frac_route in correct_full_homing&escape']

for session_id in cluster_experiments_objects:
    session = Process(session_id).load_session()
    logger.info("Loaded a session with the following details: {}".format(session_id))

    aefizz = AnalyzeEfizz(session, Settings_ae)
    aefizz.load_data(analysis_name)
    for var in variable:
        aefizz.execute(analysis_name, var)

logger.success("Efizz analysis pipeline complete")