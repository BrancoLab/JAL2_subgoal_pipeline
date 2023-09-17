"""

The point of this script is to output the preprocessing class object of choice. Either a synthetic object or a real one. Both of these classes
are defined in the pp_classes.py script. The overall pupose of this module is to output a preprocessed data object that can be used for further
analysis.

NOTE: A potential consideration is to have this script output all of the preprocessed data objects if the user wants access to all of them.

"""

# Custom libaries
from behave_analysis.postprocess.pp_classes import SyntheticDataPostprocessor, DataPostprocessor
from settings.settings_postprocess import defined_settings_postprocess

# Open source libaries
from loguru import logger
import pickle
import os
import dill as dill_pickle

class Postprocessor:
    """
    The responsibility of this class is to output and save a postprocessed object as a pickle file in the processed data folder.
    """
    
    def __init__(self, session):
        self.session = session
        self.tracking_data = self.open_tracking_data()
        preprocessObect = self.output_postprocessing_class(settings = defined_settings_postprocess)
        self.save_postprocessed_object(preprocessObect, settings = defined_settings_postprocess)
        
    def open_tracking_data(self):
        """ 
        This function opens the tracking data and appends it to the postprocessing object.
        
        NOTE: This function consider whether this function is out of place. 
        It is not really the responsibility of the postprocessing class to open the tracking data.
        """
        
        file = os.path.join(self.session.processed_path, "fully_processed_tracking_data.pickle")
        
        try:
            with open(file, "rb") as dill_file:
                tracking_data = dill_pickle.load(dill_file)
        
        except FileNotFoundError:
            logger.error(f"Tracking data not found for session: {self.session.name}")
            raise FileNotFoundError
        
        return tracking_data
    
    def output_postprocessing_class(self, settings) -> object:
        """ 
        Depending on the user defined settings, this function will output a postprocessing object of choice. Either a synthetic object or a real one.
        """
        
        if 'synthetic' in settings.cluster_type:
            postprocessObject = SyntheticDataPostprocessor(cluster_labels_to_filter = settings.cluster_type,
                                                          tracking_data = self.tracking_data,
                                                          session = self.session)
                    
        elif settings.cluster_type in ['all', 'good', 'mua', 'noise']:
            postprocessObject = DataPostprocessor(cluster_labels_to_filter = settings.cluster_type,
                                                 tracking_data = self.tracking_data,
                                                 session = self.session)
        
        return postprocessObject
            
    def save_postprocessed_object(self, postprocessObject, settings) -> None:
        """
        A function that saves the postprocessed object to a pickle file in the processed data folder.
        """
        
        try: 
            fileObj = open(self.session.processed_path + "\\" + "postprocessclass" + "_" + str(settings.cluster_type), 'wb')
            pickle.dump(postprocessObject, fileObj)
            fileObj.close()
            logger.success(f"The postprocessing of the data has finished and the postprocessed object has been saved to a pickle file")
        
        except:
            logger.error("Didn't work did it")
            raise Exception