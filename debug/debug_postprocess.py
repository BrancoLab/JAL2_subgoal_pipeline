from behave_analysis.postprocess.pp_main import Postprocessor
from loguru import logger
from databank import experiments_objects
from behave_analysis.process.process import Process

def postprocess():
    """ 
    A function that outputs and saves a postprocessed object as a pickle file in the processed data folder.
    """
    logger.info("The post processing of the data has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        Postprocessor(session)
    logger.success("The post processing of the data has finished and the postprocessed object has been saved to a pickle file")
    
postprocess()