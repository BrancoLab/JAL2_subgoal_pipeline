"""A script to load the significant clusters for the TunED analysis 
such that the null distribution is only generated on significant Tuned cells."""

# standard libaries
import os

# third party libaries
import polars as pl

class ReturnSigClusters:
    """Return a filtered rayleigh df with only significant hdir clusters"""
    def __init__(self, post_process_object, settings):
        self.rayleigh_path = self.get_rayleigh_path(post_process_object)
        self.rayleigh_df = self.load_rayleigh_arrow_file(self.rayleigh_path, settings)
        self.sig_clusters = self.return_significant_clusters(self.rayleigh_df)

    def get_rayleigh_path(self, post_process_object) -> str:
        """Retreive the path to rayleigh test results"""
        base = post_process_object.session.base_path
        processed = post_process_object.session.processed_path
        return os.path.join(base, processed, "models", "Rayleigh")


    def load_rayleigh_arrow_file(self, rayleigh_path, settings) -> pl.DataFrame:
        """Load hdir rayleigh results assuming they are most tuned"""
        path = os.path.join(rayleigh_path, settings.cluster_type[0], "all_time", "hdir_Rayleigh.arrow")
        return pl.read_ipc(path)


    def return_significant_clusters(self, rayleigh_results_df) -> pl.DataFrame:
        """Return hdir cluster IDs that are sig in atleast one compartment"""
        return rayleigh_results_df.filter(pl.col('Rayleigh_sig').arr.contains(1))