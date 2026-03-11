from loguru import logger

from behave_analysis.postprocess.pp_main import Postprocessor
from behave_analysis.process.process import Process
from databank import experiments_objects

def postprocess():
    """ 
    A function that outputs and saves a postprocessed object as a pickle file in the processed data folder.
    """
    logger.info("The post processing of the data has started")
    for session_id in experiments_objects:
        session = Process(session_id).load_session()
        logger.info("Loaded a session with the following details: {}".format(session_id))
        Postprocessor(session)
    logger.success("The post processing of the data has finished and the postprocessed object has been saved to a pickle file")
    
postprocess()
