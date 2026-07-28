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

for session_id in cluster_experiments_objects:
	logger.info("Creating a session with the following details: {}", session_id)
	process_object = Process(session_id)
	process_object.create_session(settings_process)

for session_id in cluster_experiments_objects:
	session = Process(session_id).load_session()
	logger.info("Loaded a session with the following details: {}", session_id)
	Postprocessor(session)