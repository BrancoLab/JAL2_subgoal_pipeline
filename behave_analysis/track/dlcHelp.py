import glob
import pandas as pd
import os
from loguru import logger
import yaml
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import scipy.ndimage

# Custom
from behave_analysis.track.kalmanFilter import kalmann

class DLC:
    """A class to handle the DLC tracking data. This class is used to extract the tracking data 
    from the DLC outputted .h5 file and save it to a dictionary. The class also creates a 3D array 
    of tracking data from DLC of length number of frames. The main functions are then to
    process poor tracking data.
    
    Main refactor to consider is to use a kalman filter for positional tracking but 
    will focus on other things as this seems like large task"""
    
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
        """A function that creates an array of shape (number of frames, number of body parts, 3)
        where the 3 is for x, y, and likelihood. A potential refactor would be to covert into a dictionary
        where there are more clear defined keys e.g. Leave for now.
        """
        self.tracking_data_array = np.zeros((session.video.num_frames, len(self.tracking_data['bodyparts']), 3))
        
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
            
    def remove_bad_tracking_data(self, session):
        self.correct_out_of_frame_tracking(session)
        # self.replace_low_confidence_points_with_nan() - Removing this as it is not needed with the kalman filter
        # self.interpolate_nan_values() # Remove this as it is not needed with the kalman filter
        # self.apply_median_filter(filter_length = 7) # Old smoothing function replaced with kalman filter
        self.replace_points_far_from_median_bodypart_with_nan()
        # self.interpolate_nan_values() - Note needed with kalman filter     
            
    def correct_out_of_frame_tracking(self, session):
        self.tracking_data_array[self.tracking_data_array < 0] = 0
        self.tracking_data_array[:,:,0][self.tracking_data_array[:, :, 0] > (session.video.width-1)]  = session.video.width - 1
        self.tracking_data_array[:,:,1][self.tracking_data_array[:, :, 1] > (session.video.height-1)] = session.video.height - 1
        
    def replace_low_confidence_points_with_nan(self) -> None:
        """If the confidence score for a point is below the threshold set in the settings_track file, 
        then replace the likelihood with a nan. Log to the user how many points were replaced."""
        
        low_confidence_points = self.tracking_data_array[:, :, 2] < self.settings.min_confidence_in_tracking
        self.tracking_data_array[low_confidence_points, :2] = np.nan
        
        numOflowConfidencePoints = np.count_nonzero(low_confidence_points)
        numOfTotalPoints = low_confidence_points.shape[0] * low_confidence_points.shape[1]
        perct = numOflowConfidencePoints / numOfTotalPoints
        logger.warning(f"Replaced {numOflowConfidencePoints} out of {numOfTotalPoints} points ({perct:.2f}) with nan due not being above the confidence threshold of {self.settings.min_confidence_in_tracking}")
    
    def interpolate_nan_values(self):
        """Use numpy to interpolate the nan values in the tracking data. From last confident point to next confident point, intepolate all nans between"""
        
        for i, _ in enumerate(self.tracking_data['bodyparts']):
            self.tracking_data_array[:, i, :2] = np.array(pd.DataFrame(self.tracking_data_array[:, i, :2]).interpolate().fillna(method='bfill').fillna(method='ffill'))
    
    def apply_median_filter(self, filter_length=7):
        """Apply a median filter to the tracking data to remove outliers. A median filter is a non-linear filter that is commonly used to remove noise from an image or a signal. 
        The filter works by replacing each element in the signal with the median value of its neighboring pixels or elements."""
        
        self.tracking_data_array[:, :, :2] = scipy.ndimage.median_filter(self.tracking_data_array[:, :, :2], size=(filter_length, 1, 1), mode='nearest')
    
    def replace_points_far_from_median_bodypart_with_nan(self):
        median_position_across_bodyparts = np.nanmedian(self.tracking_data_array[:, :, :2], axis=1) 
        distance_from_median_position = ((self.tracking_data_array[:, :, 0] - median_position_across_bodyparts[:, 0:1])**2 + \
                                         (self.tracking_data_array[:, :, 1] - median_position_across_bodyparts[:, 1:2])**2)**.5
        self.tracking_data_array[distance_from_median_position > self.settings.max_deviation_from_rest_of_points, :2] = np.nan
    
    def apply_kalman(self, session):
        """
           The kalman filter is a recursive algorithm that estimates the state of a system using a sequence of measurements.
           This function requires the tracking data to be in the form of a numpy array with the following dimensions:
            + (2, frames)
           The algorithm works on a single body part and thus needs to be called in a recursive manner. 
        """
        savePath = os.path.join(session.file_path, "kalman_tracking_data.pickle")
        try:
            with open(savePath, 'rb') as f:
                my_dict = pickle.load(f)
                logger.info("Loaded previous pickled kalman tracking data, mmmm pickles.")
            
        except FileNotFoundError:
            logger.info("No pickled kalman tracking data found. Creating new kalman tracking data.")
            
            ldsResults = {}
            
            for i, bodypart in enumerate(self.tracking_data['bodyparts']):
                x, y = np.transpose(self.tracking_data_array[:, i, 0]), np.transpose(self.tracking_data_array[:, i, 1])
                xy = np.vstack((x, y))
                results = kalmann(xy)
                ldsResults[bodypart] = {"x": results["x"], 
                                        "y": results["y"], 
                                        "likelihood": self.tracking_data_array[:, i, 2],
                                        "xVelocity": results["xVelocity"],
                                        "yVelocity": results["yVelocity"],
                                        "xAccel": results["xAccel"],
                                        "yAccel": results["yAccel"],
                                        }
                 
                self.lds_tracking_data = ldsResults
                self.save_kalman(self.lds_tracking_data, session)
        
    def save_kalman(self, dictionary, session):
        """
           Save the kalman tracking dictionary to a pickle file contained within the session folder. 
        """
        savePath = os.path.join(session.file_path, "kalman_tracking_data.pickle")
        with open(savePath, "wb") as dill_file: 
            pickle.dump(dictionary, dill_file)