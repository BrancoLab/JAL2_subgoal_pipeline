# Custom lib
from behave_analysis.utils.AI_dataClass_objects import Elecetrophysiology

# OS libs
import os
import numpy as np

class LoadEfizz():
    def __init__(self, session_ID):
        self.file_path = session_ID.file_path
        self.files = self.collect_efizz_files()
        self.select_and_load_efizz_files()
        
    def collect_efizz_files(self) -> list:
        """
        A function that collects all the efizz files from the session folder. Although dirnames
        not used, it is needed to walk the directory tree
        """
        files = []
        for dirpath, dirnames, filenames in os.walk(self.file_path):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files
    
    def select_and_load_efizz_files(self) -> None:
        """
        A function that selects the efizz files that are needed for the pipeline.
        Should return a list of strings
        """
        try:
            self.spike_times = np.load(self.filter_by_ending(self.files, "spike_times.npy")[0])
            self.spike_clusters = np.load(self.filter_by_ending(self.files, "spike_clusters.npy")[0])
            self.TTL_bin_path = self.filter_by_ending(self.files, "imec0.ap.bin")[0]
            return Elecetrophysiology(spike_times = self.spike_times, 
                                      spike_clusters = self.spike_clusters,
                                      TTL_bin_path = self.TTL_bin_path)
            
        except IndexError:
            raise IndexError(f"One of these files did not exsist within {self.files}")
        
    def filter_by_ending(self, lst, ending):
        """
        Returns a list of strings from `lst` that end with the specified `ending`.
        """
        return [s for s in lst if s.endswith(ending)]
        
      