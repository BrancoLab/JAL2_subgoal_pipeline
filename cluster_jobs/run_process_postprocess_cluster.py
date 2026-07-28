import sys
from pathlib import Path

# Ensure repository-root imports (e.g. databank, run, settings) work when this
# script is invoked as a file path from cluster jobs.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from databank import cluster_experiments_objects
from run.run_process import process
from run.run_postprocess import postprocess
from settings.settings_process import settings_process

process(cluster_experiments_objects, settings_p=settings_process)

postprocess(cluster_experiments_objects)