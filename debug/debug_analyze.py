# OS Libaries
from loguru import logger

# Custom Libaries
from databank import experiments_objects
from behave_analysis.process.process import Process
from behave_analysis.analyze.analyze_efizz import AnalyzeEfizz

def analyze():
    """
    A function that calls all the analysis modules and is designed to be run last and for the whole dataset.
    """
    logger.info("The analysis pipeline has started")
    for session_ID in experiments_objects:
        session = Process(session_ID).load_session()
        AnalyzeEfizz(session = session)
    
    logger.success("Analysis pipeline complete")
    
analyze()
