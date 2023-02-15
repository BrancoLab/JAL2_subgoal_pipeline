import glob
import pandas as pd
import os
from loguru import logger
import yaml
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt

class DLC:
    """A class to handle the DLC tracking data. This class is used to extract the tracking data from the DLC outputted .h5 file 
    and save it to a dictionary. The class also creates a 3D array of tracking data from DLC of length number of frames. 
    Not exactly sure what is going on here, need to look into it more."""
    
    def run_deeplabcut_tracking(self, session):
        """Check if DLC has been run on a video before, if not run analyze videos. If a DLC
        file already exists don't run DLC again. It requires the DLC settings file
        to contain resnet in its name which may not be the case for all DLC models
        and versions.

        Args:
            session (object): A data class containing relevant information for tracking contained within settings_track.py
        """
        dlc_already_run = bool(glob.glob(os.path.join(session.file_path, "*resnet*"))) # Does a file exist with this token in the name?
        
        if dlc_already_run:
            logger.info("DeepLabCut has already been run for this session: {} - {}".format(session.number, session.name))
            
        else:
            logger.info("Running DeepLabCut tracking for session: {} - {}".format(session.number, session.name))
            from deeplabcut.pose_estimation_tensorflow import analyze_videos
            analyze_videos(self.settings.dlc_settings_file, session.video.video_file)
    
    def extract_data_from_dlc_file(self, session) -> None:
        """Ingests a H5 file outputted from DLC analysis, body parts, and
        model name. Changing the string in dlc network name maybe necessary if using different model
        type. 
        
        The function saves the body parts tracked by DLC to the tracking data dictionary.

        Args:
            session (obejct): Session settings object data class
        """
        
        # Load DLC Tracking Data
        dlc_tracking_file = glob.glob(os.path.join(session.file_path, "*.h5"))[0] #Selects the .h5 file in video dir
        self.dlc_output = pd.read_hdf(dlc_tracking_file) #Converts .h5 to pandas
        
        # Load DLC Config from settings dataclass 
        with open(self.settings.dlc_settings_file) as file: 
            dlc_settings = yaml.safe_load(file)
        
        # Extract body parts and model name
        self.tracking_data['bodyparts'] = dlc_settings['bodyparts']
        logger.info(f"The bodyparts tracked by DLC are: {self.tracking_data['bodyparts']}")
        self.dlc_network_name = dlc_tracking_file[dlc_tracking_file.find('DLC_resnet'):-3] # This line breaks if different model names are used
        assert self.dlc_network_name, "No DLC name found, has a different model been used?"
        logger.info(f"The DLC network name is: {self.dlc_network_name}")

        return None
    
    def create_array_with_dlc_tracking_data(self, session) -> None:
        """A function that creates a 3D array of tracking data from DLC of length
        number of frames. Not exactly sure what is going on here, need to look into it
        more. 

        Args:
            session (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.tracking_data_array = np.zeros((session.video.num_frames, 
                                             len(self.tracking_data['bodyparts']), 
                                             3))
        
        for i, body_part in enumerate(self.tracking_data['bodyparts']):
            for j, axis in enumerate(['x', 'y']):
                self.tracking_data_array[:, i, j] = self.dlc_output[self.dlc_network_name][body_part][axis].values
            self.tracking_data_array[:, i, 2] = self.dlc_output[self.dlc_network_name][body_part]['likelihood'].values
        
        return None
    
    def create_dlc_tracking_array(self, session) -> None:
        """Create and fill an array of tracking data from DLC.

        Args:
            session (object): session dataclass
        """
        self.tracking_data = {}
        self.extract_data_from_dlc_file(session)
        self.create_array_with_dlc_tracking_data(session)
        
        return None
    
    def save_tracking(self, session):
        with open(session.video.tracking_data_file, "wb") as dill_file: 
            pickle.dump(self.tracking_data, dill_file)
    
    def plot_tracking(self):
        if self.settings.display_tracking_output:
            for axis in [0,1]:
                plt.figure()
                plt.title('Example of 10,000 time-points of tracking data - axis {}'.format(axis))
                for bodypart in self.tracking_data['bodyparts']:
                    plt.plot(self.tracking_data[bodypart][10000:20000, axis])
                plt.legend(self.tracking_data['bodyparts'])
            plt.figure(figsize=(12,6))
            plt.title('Histogram of confidence in tracking data')
            plt.hist(self.tracking_data_array[:,:,2], 20, density=True)
            plt.show()