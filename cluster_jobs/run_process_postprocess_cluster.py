from loguru import logger

from databank import cluster_experiments_objects
from run.run_process import process
from run.run_postprocess import postprocess
from settings.settings_process import settings_process

process(cluster_experiments_objects, settings_p=settings_process)

postprocess(cluster_experiments_objects)