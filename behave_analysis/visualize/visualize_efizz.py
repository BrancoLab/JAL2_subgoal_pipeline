# OS libaries
from loguru import logger
import polars as pl
import os
import numpy as np

# Import custom settings
from settings.settings_visualize import defined_settings_visualize as settings_v
from behave_analysis.visualize.efizz.egocentric_firing_map_binned import egocentric_firing_map
from behave_analysis.visualize.efizz.rayleigh_map import rayleigh_map
from behave_analysis.visualize.efizz.stim_resp_functions import single_cluster_raster, rasters, PSTH_all_neurons, PSTH_single_neurons
from behave_analysis.visualize.efizz.spatial_tuning_functions import spatial_position_firing, spatial_position_firing_hdir
from behave_analysis.visualize.efizz.heatmap import single_unit_level_heatmaps
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.visualize.visualize_utils import open_postprocess_object
from behave_analysis.analyze.filtering_data.filtering_functions import extract_all_or_custom_conditions


class Visualize_efizz:
    """A class that creates visualizations on the efizz data with no stats or analysis.
    The purpose of this class is to create visualizations of the efizz data for the user to inspect
    in order to understand the data better and aid in the analysis process.

    TODO:
        This class has been half refactored to use a split up postprocess object. The other half needs to be refactored.
        The video_spike_count_df and video_df have been split but there are still some data attached to the pp object that
        requires it to be loaded in and it's very slow as can be 50-100GB. This needs to be fixed.

    """

    def __init__(self, session):
        self.session = session
        # load in processed data
        self.processed_data = open_postprocess_object(self.session, settings_v.cluster_type)
        base_path = os.path.join(self.session.base_path, self.session.processed_path)
        self.video_df = pl.read_csv(os.path.join(base_path, "full_video_dataframe.csv"))
        self.video_spike_count_df = pl.read_parquet(
            os.path.join(base_path + "\\" + str(settings_v.cluster_type) + "_video_spike_count_df.parquet"), 
            low_memory=True,
            use_pyarrow = True,
            memory_map=True,
        )
        logger.info("Visualize_efizz class initialized - Time to plot some efizz!")

    ##----------TUNING PLOTTING
    def run_tuning_functions(self):
        """Excute five visulisation functions on the efizz data related to spatial information

        Responsibilities:
            Create spatial firing maps for each neuron coloured by head direction
            Create spatial firing maps for each neuron with no colouring
            Create egocentric firing maps for each neuron
            Create rayleigh maps for each neuron
            Create population level heatmaps

        Save the resulting plots to a spatial_firing directory in the processed folder
        """
        logger.info("Starting to make some efizz tuning plots...")
        spatial_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "spatial_firing"))

        # Make single unit heatmaps per condition
        single_unit_level_heatmaps(
            video_and_spike_data=self.video_spike_count_df,
            conditions=extract_all_or_custom_conditions(settings_v, self.session),
            save_base=spatial_path,
            session=self.session,
        )

        # where a neuron fires coloured by hdir
        save_path = make_directory(os.path.join(spatial_path, "spatial_firing_hdir_color", self.processed_data.select_clusters))
        spatial_position_firing_hdir(
            data=self.processed_data.spike_data,
            clu_label=self.processed_data.clu_label,
            video_df=self.video_df,
            save_path=save_path + "/" + self.processed_data.select_clusters,
            show_plots=settings_v.show_plots,
        )

        # where a neuron fires
        save_path = make_directory(os.path.join(spatial_path, "spatial_firing_maps", self.processed_data.select_clusters))
        spatial_position_firing(
            data=self.processed_data.spike_data,
            clu_label=self.processed_data.clu_label,
            video_spike_count_df=self.video_spike_count_df,
            save_path=save_path + "/" + self.processed_data.select_clusters,
            show_plots=settings_v.show_plots,
        )

        cluster_Ids = self.processed_data.clu_label["spike_clusters"].unique().to_numpy()

        # egocentric view of features where a neuron fires
        # TODO ego firing map failed for me (laurence) need to debug
        # egocentric_firing_map(
        #     self.processed_data.frame_by_cluster_matrix,
        #     self.video_df,
        #     self.processed_data.clu_label,
        #     self.session,
        #     conditions=extract_all_or_custom_conditions(settings_v, self.session),
        #     cluster_Ids=cluster_Ids[cluster_Ids > 0],
        #     settings=settings_v,
        # )

        # a map of where rayleighs point to
        rayleigh_map(
            self.processed_data.frame_by_cluster_matrix,
            self.video_df,
            self.processed_data.clu_label,
            self.session,
            conditions=extract_all_or_custom_conditions(settings_v, self.session),
            cluster_Ids=cluster_Ids[cluster_Ids > 0],
            settings=settings_v,
            tracking_data=self.processed_data.tracking_data,
        )
        logger.info(f"Finished! Making some efizz tuning plots...")

    ##------------STIMULUS RESPONSE PLOTTING
    def run_stim_resp_plotting(self):
        """Make plots of stimulus response"""
        logger.info(f"Starting to make some plots of threat stimulus responses.")
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "raster"))
        rasters(
            data=self.processed_data.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + self.processed_data.select_clusters + "_cluster_raster_trial_" + str(settings_v.stim_type) + ".png",
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "PSTH"))
        PSTH_all_neurons(
            session=self.session,
            data=self.processed_data.spike_data,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path)
            + "/"
            + self.processed_data.select_clusters
            + "_clusters_PSTH_all_neurons_"
            + str(settings_v.stim_type)
            + ".png",
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "PSTH_single_cluster"))
        PSTH_single_neurons(
            data=self.processed_data.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + str(settings_v.stim_type) + "_single_" + self.processed_data.select_clusters,
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "raster_single_cluster"))
        single_cluster_raster(
            data=self.processed_data.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + self.processed_data.select_clusters + "_clusters_" + str(settings_v.stim_type),
        )
