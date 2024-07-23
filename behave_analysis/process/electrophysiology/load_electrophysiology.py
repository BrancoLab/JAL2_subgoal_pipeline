# Custom lib
from behave_analysis.utils.AI_dataClass_objects import Electrophsyiology

# OS libs
import os
import numpy as np
from loguru import logger


class LoadEfizz:
    def __init__(self, session_ID):
        self.file_path = os.path.join(session_ID.base_path, session_ID.file_path)
        self.files = self.collect_efizz_files()
        self.select_and_load_efizz_files()

    def collect_efizz_files(self) -> list:
        """
        A function that collects all the efizz files from the session folder. Although dirnames
        not used, it is needed to walk the directory tree
        """
        logger.info("Searching through efizz files")
        efizz_folder = os.path.join(self.file_path,[x for x in os.listdir(self.file_path) if '_g0' in x][0])
        files = []
        for dirpath, dirnames, filenames in os.walk(efizz_folder):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        logger.success("Efizz file names collected")
        return files

    def select_and_load_efizz_files(self) -> None:
        """
        A function that selects the efizz files that are needed for the pipeline.
        Should return a list of strings
        """
        try:
            assert len(self.files) != 0, "Session list should not be empty"
            assert len(self.filter_by_ending(self.files, "spike_times.npy")) == 1, "There should only be one spike_times.npy file"
            assert len(self.filter_by_ending(self.files, "spike_clusters.npy")) == 1, "There should only be one spike_clusters.npy file"
            assert len(self.filter_by_ending(self.files, "cluster_group.tsv")) == 1, "There should only be one cluster_group.tsv file"

            self.spike_times = np.load(self.filter_by_ending(self.files, "spike_times.npy")[0])
            logger.info(f"The number of spikes is: {len(self.spike_times)}")
            self.spike_clusters = np.load(self.filter_by_ending(self.files, "spike_clusters.npy")[0])
            self.spike_clusters = np.hstack(self.spike_clusters)
            sync = self.filter_by_ending(self.files, "exported.imec0.ap.bin")
            if len(sync) > 0: 
                self.imec_sync_path = sync[0]
            else:
                logger.warning("No exported .bin sync channel was found!")
                self.imec_sync_path = self.filter_by_ending(self.files, "_t0.imec0.ap.bin")[0]
            self.imec_bin_path = self.filter_by_ending(self.files, "_t0.imec0.ap.bin")[0]
            self.cluster_group = np.loadtxt(self.filter_by_ending(self.files, "cluster_group.tsv")[0], delimiter="\t", skiprows=1, dtype=str)
            self.num_of_good_units = self.count_number_of_label_units("good")
            logger.info(f"The number of good units is: {self.num_of_good_units} out of {len(self.cluster_group)} units")
            num_mua = self.count_number_of_label_units("mua")
            logger.info(f"The number of mua is: {num_mua} out of {len(self.cluster_group)} units")

            # assert self.cluster_group[0][0] == "0", "The first cluster should be indexed by 0"  # sort check

            return Electrophsyiology(
                spike_times=self.spike_times,
                spike_clusters=self.spike_clusters,
                cluster_group=self.cluster_group,
                TTL_bin_path=self.imec_bin_path,
                number_of_good_units=self.num_of_good_units,
            )

        except IndexError:
            raise IndexError(f"One of these files did not exsist within {self.files}")

    def filter_by_ending(self, lst, ending):
        """
        Returns a list of strings from `lst` that end with the specified `ending`.
        """
        return [s for s in lst if s.endswith(ending)]

    def count_number_of_label_units(self, label):
        """
        A function that counts the number of good units in the cluster group file
        """
        return np.count_nonzero(self.cluster_group[:, 1] == label)
