# OS libaries
from loguru import logger
import polars as pl
import os
import numpy as np

# Import custom settings
from behave_analysis.analyze.analyze_efizz import merge_spike_df_video_df
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
        self.base_path = os.path.join(self.session.base_path, self.session.processed_path)
        logger.info("Visualize_efizz class initialized - Time to plot some efizz!")

    def load_data(self, visualization_name):
        
        self.spike_data = pl.read_csv(os.path.join(self.base_path, settings_v.cluster_type + "_spike_data.csv"))
        self.clu_ids = np.load(os.path.join(self.base_path, settings_v.cluster_type + "_cluster_ids.npy"))
        if hasattr(self.spike_data, 'groupby'):
            self.clu_label = self.spike_data.groupby(["spike_clusters"]).first()
        elif hasattr(self.spike_data, 'group_by'):
            self.clu_label = self.spike_data.group_by(["spike_clusters"]).first()

        if visualization_name in ["single_unit_heatmaps", "spatial_position_firing", "spatial_position_firing_hdir"]:
            self.video_df = pl.read_csv(os.path.join(self.base_path, "full_video_dataframe.csv"))
        if visualization_name in ["single_unit_heatmaps", "spatial_position_firing"]:
            video_spike_count_path = (os.path.join(self.session.base_path, self.session.processed_path)
                                        + "/"
                                        + "spike_count_by_frame_and_"
                                        + self.cluster_type
                                        + "cluster" + self.qualifier
                                        + ".csv"
                                    )
            self.spike_count_df = pl.read_csv(video_spike_count_path)
            self.video_spike_count_df = merge_spike_df_video_df(self.spike_count_df, self.video_df)
        
    ##----------TUNING PLOTTING
    def run_visualizations(self, visualization_name):
        """Excute visulisation functions on the efizz data related to spatial information and stimulus triggered responses!

        Responsibilities:
            Create spatial firing maps for each neuron coloured by head direction
            Create spatial firing maps for each neuron with no colouring
            Create egocentric firing maps for each neuron
            Create rayleigh maps for each neuron
            Create population level heatmaps
            Rasters and PSTHs for each neuron and population level in response to threat

        Save the resulting plots to a spatial_firing or stim_resp directory in the processed folder
        """
        spatial_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "spatial_firing"))
        stim_resp_path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "stim_resp"))

        if visualization_name == "single_unit_heatmaps":
            # Make single unit heatmaps per condition
            single_unit_level_heatmaps(
                video_and_spike_data=self.video_spike_count_df,
                conditions=extract_all_or_custom_conditions(settings_v, self.session),
                save_base=spatial_path,
                session=self.session,
                )

        if visualization_name == "spatial_position_firing_hdir":
            # where a neuron fires coloured by hdir
            save_path = make_directory(os.path.join(spatial_path, "spatial_firing_hdir_color", settings_v.cluster_type))
            spatial_position_firing_hdir(
                data=self.spike_data,
                clu_label=self.clu_label,
                video_df=self.video_df,
                save_path=save_path + "/" + settings_v.cluster_type,
                show_plots=settings_v.show_plots,
                )

        if visualization_name == "spatial_position_firing":
            # where a neuron fires
            save_path = make_directory(os.path.join(spatial_path, "spatial_firing_maps", settings_v.cluster_type))
            spatial_position_firing(
                data=self.spike_data,
                clu_label=self.clu_label,
                video_spike_count_df=self.video_spike_count_df,
                save_path=save_path + "/" + settings_v.cluster_type,
                show_plots=settings_v.show_plots,
                )
        
        if visualization_name == "pop_rasters":
            rasters(
            data=self.spike_data,
            session=self.session,
            stim_type=settings_v.stim_type,
            show_plots=settings_v.show_plots,
            save_path=str(os.path.join(stim_resp_path, "raster")) + "/" + settings_v.cluster_type + "_cluster_raster_trial_" + str(settings_v.stim_type) + ".png",
        )

        if visualization_name == "pop_PSTH":
            PSTH_all_neurons(
                session=self.session,
                data=self.spike_data,
                stim_type=settings_v.stim_type,
                show_plots=settings_v.show_plots,
                save_path=str(os.path.join(stim_resp_path, "PSTH"))
                + "/"
                + settings_v.cluster_type
                + "_clusters_PSTH_all_neurons_"
                + str(settings_v.stim_type)
                + ".png",
            )

        if visualization_name == "PSTH_single_cluster":
            PSTH_single_neurons(
                data=self.spike_data,
                session=self.session,
                stim_type=settings_v.stim_type,
                show_plots=settings_v.show_plots,
                save_path=str(os.path.join(stim_resp_path, "PSTH_single_cluster")) + "/" + str(settings_v.stim_type) + "_single_" + settings_v.cluster_type,
            )

        if visualization_name == "single_cluster_raster":
            single_cluster_raster(
                data=self.spike_data,
                session=self.session,
                stim_type=settings_v.stim_type,
                show_plots=settings_v.show_plots,
                save_path=str(os.path.join(stim_resp_path, "raster_single_cluster")) + "/" + settings_v.cluster_type + "_clusters_" + str(settings_v.stim_type),
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