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

assert len(cluster_experiments_objects) != 0, "Session list should not be empty"

def process(experiments_objects, settings_p):
	for session_id in experiments_objects:
		logger.info(f"Processing session: {session_id.nick_name} on {session_id.experiment_date} for experiment: {session_id.experiment_name}")
		process_object = Process(session_id)
		process_object.create_session(settings_p)
	logger.success("Processing complete")

def postprocess(experiments_objects):
	for session_id in experiments_objects:
		session = Process(session_id).load_session()
		logger.info(f"Post-processing session: {session_id.nick_name} on {session_id.experiment_date} for experiment: {session_id.experiment_name}")
		Postprocessor(session)
	logger.success("Postprocessing complete")

# analyze efizz data
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz
from settings.settings_analyze_efizz import Settings_ae

def analyze_efizz(analysis_name = None, variable = None, experiments_objects = []):
	for session_id in experiments_objects:
		session = Process(session_id).load_session()
		logger.info(f"Running analyses for session: {session_id.nick_name} on {session_id.experiment_date} for experiment: {session_id.experiment_name}")

		aefizz = AnalyzeEfizz(session, Settings_ae)
		aefizz.load_data(analysis_name)
		if variable is None:
			aefizz.execute(analysis_name)
		else:
			for var in variable:
				aefizz.execute(analysis_name, var)

	logger.success("Efizz analysis pipeline complete")

def main():

	# process(cluster_experiments_objects, settings_p=settings_process)

	# postprocess(cluster_experiments_objects)

	analysis_name='EscapePattern'
	variable = ['residual: frac_route in homing&escape - 2D_position in explore',
			 	'residual: frac_route in homing&escape - bird_dist_shelter in explore',
			 	'residual: frac_route in homing&escape - speed in explore',
			 	'residual: frac_route in homing&escape - distance_shelter in explore',]

	analyze_efizz(analysis_name, variable, experiments_objects=cluster_experiments_objects)

if __name__ == "__main__":
	main()