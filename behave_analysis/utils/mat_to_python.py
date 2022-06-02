"""Often fellow lab members will work in MatLab and save their output in MatLab Structs.
This script is to help convert that struct away from nasty MatLab into Python
Input: file path of .mat file which is a struct
Returns: A python dictionary"""

# OS libaries
from scipy.io import loadmat
import scipy
import mat73
from loguru import logger

class convert_matlab_struct:
    def __init__(self, file_path_of_mat):
        logger.info("Converting your .mat into a Python Dictionary..")
        self.dictionary = self.convert_mat_file(file_path_of_mat)
        
    def convert_mat_file(self, filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = mat73.loadmat(filename, use_attrdict=True)
        return data