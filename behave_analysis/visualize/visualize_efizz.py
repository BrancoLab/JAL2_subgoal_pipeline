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
        base_path = os.path.join(self.session.base_path, self.session.processed_path)

        try:
            self.spike_data = pl.read_csv(os.path.join(base_path, settings_v.cluster_type + "_spike_data.csv"))
            self.video_df = pl.read_csv(os.path.join(base_path, "full_video_dataframe.csv"))
            self.clu_ids = np.load(os.path.join(base_path, settings_v.cluster_type + "_cluster_ids.npy"))
            self.clu_label = self.spike_data.groupby(["spike_clusters"]).first()
            self.video_spike_count_df = pl.read_parquet(
                os.path.join(base_path + "\\" + str(settings_v.cluster_type) + "_video_spike_count_df.parquet"), 
                low_memory=True,
                use_pyarrow = True,
                memory_map=True,
            )
        except FileNotFoundError:
            raise FileNotFoundError("The efizz data has not been processed yet. Please run the process pipeline first for all files.")
        
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
        save_path = make_directory(os.path.join(spatial_path, "spatial_firing_hdir_color", settings_v.cluster_type))
        spatial_position_firing_hdir(
            data=self.spike_data,
            clu_label=self.clu_label,
            video_df=self.video_df,
            save_path=save_path + "/" + settings_v.cluster_type,
            show_plots=settings_v.show_plots,
        )

        # where a neuron fires
        save_path = make_directory(os.path.join(spatial_path, "spatial_firing_maps", settings_v.cluster_type))
        spatial_position_firing(
            data=self.spike_data,
            clu_label=self.clu_label,
            video_spike_count_df=self.video_spike_count_df,
            save_path=save_path + "/" + settings_v.cluster_type,
            show_plots=settings_v.show_plots,
        )

        # TODO: the two plots below are very nice but ver very slow to make
        # egocentric view of features where a neuron fires
        cluster_Ids = self.clu_label["spike_clusters"].unique().to_numpy()
        # egocentric view of features where a neuron fires
        # TODO ego firing map failed for me (laurence) need to debug
        # egocentric_firing_map(
        #     self.processed_data.frame_by_cluster_matrix,
        #     self.video_df,
        #     self.clu_label,
        #     self.session,
        #     conditions=extract_all_or_custom_conditions(settings_v, self.session),
        #     cluster_Ids=cluster_Ids[cluster_Ids > 0],
        #     settings=settings_v,
        # )

        # # a map of where rayleighs point to
        # rayleigh_map(
        #     self.processed_data.frame_by_cluster_matrix,
        #     self.video_df,
        #     self.clu_label,
        #     self.session,
        #     conditions=extract_all_or_custom_conditions(settings_v, self.session),
        #     cluster_Ids=cluster_Ids[cluster_Ids > 0],
        #     settings=settings_v,
        #     tracking_data=self.processed_data.tracking_data,
        # )
        # logger.info(f"Finished! Making some efizz tuning plots...")

    ##------------STIMULUS RESPONSE PLOTTING
    def run_stim_resp_plotting(self):
        """Make plots of stimulus response"""
        logger.info(f"Starting to make some plots of threat stimulus responses.")
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "raster"))
        rasters(
            data=self.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + settings_v.cluster_type + "_cluster_raster_trial_" + str(settings_v.stim_type) + ".png",
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "PSTH"))
        PSTH_all_neurons(
            session=self.session,
            data=self.spike_data,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path)
            + "/"
            + settings_v.cluster_type
            + "_clusters_PSTH_all_neurons_"
            + str(settings_v.stim_type)
            + ".png",
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "PSTH_single_cluster"))
        PSTH_single_neurons(
            data=self.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + str(settings_v.stim_type) + "_single_" + settings_v.cluster_type,
        )
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp", "raster_single_cluster"))
        single_cluster_raster(
            data=self.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(stim_resp_path) + "/" + settings_v.cluster_type + "_clusters_" + str(settings_v.stim_type),
        )
