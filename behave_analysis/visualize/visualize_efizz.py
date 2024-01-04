# OS libaries
from loguru import logger
import polars as pl
import os

# Import custom settings
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.visualize.efizz.egocentric_firing_map_binned import egocentric_firing_map
from behave_analysis.visualize.efizz.stim_resp_functions import single_cluster_raster, rasters, PSTH_all_neurons, PSTH_single_neurons
from behave_analysis.visualize.efizz.tuning_functions import spatial_position_firing, spatial_position_firing_hdir
from behave_analysis.utils.creating_directories import make_directory
class Visualize_efizz:
    """
    A class for some sanity check efizz plots using kilosort clusters
    """
    
    def __init__(self,  Processed_data_object, session):
       
       self.session = session

       # load in processed data
       self.processed_data = Processed_data_object
       self.video_df = pl.read_csv(
            os.path.join(self.session.base_path, self.session.processed_path, "full_video_dataframe.csv")
        )
       
       logger.info("Visualize_efizz class initialized - Time to plot some efizz!")

    def run_tuning_functions(self):
        """Make tuning plots"""
        logger.info(f"Starting to make some efizz tuning plots...")
        spatial_path = os.path.join(self.session.base_path,self.session.processed_path, 'spatial_firing')
        make_directory(spatial_path)
        # where a neuron fires coloured by hdir
        make_directory(os.path.join(spatial_path, 'spatial_firing_hdir_color',self.processed_data.select_clusters))
        spatial_position_firing_hdir(data = self.processed_data.spike_data, 
                                     clu_label = self.processed_data.clu_label, 
                                     video_df = self.video_df,
                                     save_path = os.path.join(spatial_path, 'spatial_firing_hdir_color',self.processed_data.select_clusters) + "/" + self.processed_data.select_clusters,
                                     show_plots = settings_v.show_plots)
        make_directory(os.path.join(spatial_path, 'spatial_firing_maps',self.processed_data.select_clusters))
        spatial_position_firing(data = self.processed_data.spike_data, 
                                clu_label = self.processed_data.clu_label, 
                                video_spike_count_df = self.processed_data.video_spike_count_df,
                                save_path = os.path.join(spatial_path, 'spatial_firing_maps',self.processed_data.select_clusters) + "/" + self.processed_data.select_clusters,
                                show_plots = settings_v.show_plots) # ~ BUG - RuntimeError: main thread is not in main loop
        cluster_Ids = self.processed_data.video_spike_count_df["spike_clusters"].unique().to_numpy()
        egocentric_firing_map(self.processed_data.frame_by_cluster_matrix, 
                              self.video_df,
                              self.processed_data.clu_label,
                              self.session,
                              cluster_Ids = cluster_Ids[cluster_Ids > 0])
        logger.info(f"Finished! to make some efizz tuning plots...")

    def run_stim_resp_plotting(self):
        """Make plots of stimulus response"""
        logger.info(f"Starting to make some plots of threat stimulus responses.")
        stim_resp_path = os.path.join(self.session.base_path,self.session.processed_path, 'stim_resp')
        make_directory(stim_resp_path)
        rasters(data = self.processed_data.spike_data,
                session = self.session, 
                stim_type = settings_v.stim_type, 
                show_plots = settings_v.show_plots, 
                save_path = str(stim_resp_path) + "/" + self.processed_data.select_clusters + "_cluster_raster_trial_" + str(settings_v.stim_type) + ".png")
        PSTH_all_neurons(session = self.session, 
                         data = self.processed_data.spike_data, 
                         stim_type = settings_v.stim_type, 
                         show_plots = settings_v.show_plots, 
                         save_path = str(stim_resp_path) + "/" + self.processed_data.select_clusters + "_clusters_PSTH_all_neurons_" + str(settings_v.stim_type) + ".png")
        PSTH_single_neurons(data = self.processed_data.spike_data,
                            session = self.session, 
                            stim_type = settings_v.stim_type, 
                            show_plots = settings_v.show_plots, 
                            save_path = str(stim_resp_path) + "/" + str(settings_v.stim_type) + "_single_" + self.processed_data.select_clusters)
        single_cluster_raster(data = self.processed_data.spike_data,
                                session = self.session, 
                                stim_type = settings_v.stim_type, 
                                show_plots = settings_v.show_plots, 
                                save_path = str(stim_resp_path) + "/" + self.processed_data.select_clusters + "_clusters_" + str(settings_v.stim_type))   
 