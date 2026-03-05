"""A script for computing place cells!"""
import os
import numpy as np
import polars as pl
from dataclasses import asdict
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.analyze.PlaceCells.place_cell_utils import assign_positional_bins_to_frames, compute_spatial_information, create_centered_bins, smooth_maps
from behave_analysis.analyze.results_database_utils import add_run_to_database, settings_to_check, check_database_for_matched_results, generate_run_id
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.arena_plotting import Arena

class PlaceCells:
    def __init__(self, aefizz):
        self.aefizz = aefizz
        self.savepath = os.path.join(self.aefizz.session.base_path, self.aefizz.session.processed_path, "models", "place_cells")
        # Define spatial bins (e.g. 5cm x 5cm)
        self.bins = create_centered_bins(bin_size = self.aefizz.settings.place_cell_bin_size_pix)
        self.grid = (pl.DataFrame({"xbins": pl.Series("xbins", range(len(self.bins)-1))})
                .join(pl.DataFrame({"ybins": pl.Series("ybins", range(len(self.bins)-1))}), how="cross"))
        self.check_database_for_same_run()

    def preprocess_data(self):
        """This function preprocesses the video and spike data for place cell analysis. 
        It assigns spatial bins to each frame in the video and spike data, 
        and identifies valid bin pairs (bins where the mouse was present,as a proxy for the whole circular arena)."""

        # to be sure, remove rows of frames that have 0 spike_count
        # NB this is related to an error in how video_and_spike_data df was built, see ppclasses.py
        self.aefizz.video_and_spike_data = self.aefizz.video_and_spike_data.filter((self.aefizz.video_and_spike_data["spike_count"] > 0))
        # Bin the arena into spatial bins (e.g. 5cm x 5cm)
        self.binned_spike_df = assign_positional_bins_to_frames(self.aefizz.video_and_spike_data, bins=self.bins) # this gives us spike count per frame and cluster
        self.binned_vid_df = assign_positional_bins_to_frames(self.aefizz.video_df, bins=self.bins) # this is for computing occupancy
        # valid xy bin pairs (mouse was there, but spikes were not necessarily recorded) come from the video dataframe (circular arena mask)
        self.valid_pairs = (self.binned_vid_df.filter(pl.col("xbins").is_not_nan() & pl.col("ybins").is_not_nan())
                                              .with_columns(pl.col("xbins").cast(pl.Int64), pl.col("ybins").cast(pl.Int64))
                                              .select(["xbins", "ybins"])
                                              .unique()
                                              .with_columns(pl.lit(True).alias("is_valid")))
        
    def compute_place_fields_conditions(self):
        """This function identifies place cells based on their firing rate maps and mutual information with position.
        It iterates through each cell in each condition."""
        
        self.results_dict = {}
        for c in self.aefizz.all_conditions:
            print(f"Computing place fields for condition {c}")
            # 0. filter data to this condition
            filt_vid_df_c = filter_video_dataframe(self.binned_vid_df, c, outofshelter=None)
            # CAUTION: can't use filter_video_dataframe forthe spike data because the column homingPeriod doesn't match
            filt_spike_df_c = self.binned_spike_df.filter(pl.col("frames").is_in(filt_vid_df_c["frames"]))
            
            # compute place field and linear shift
            self.results_dict[c] = self.linear_shift_place_field(filt_spike_df_c, filt_vid_df_c)
    
    def compute_place_fields_cells(self, filt_spike_df, occupancy_map, valid_bins, return_rate_maps=False):
        # initialize results
        if return_rate_maps:
            rate_map = np.full((len(self.bins)-1, len(self.bins)-1, len(self.aefizz.cluster_Ids)), np.nan) # initialize rate map with nans
        spatial_info_bps = np.full(len(self.aefizz.cluster_Ids), np.nan) # spatial information in bits per spike
        # iterate over cells
        for i, n in enumerate(self.aefizz.cluster_Ids):
            cell_df = filt_spike_df.filter((filt_spike_df['spike_clusters'] == n)) # get spikes only for this cell
            if len(cell_df) == 0:
                print(f"No spikes for cell {n} in this condition, cannot compute place field")
                continue
            # 1. build spike_count maps for each cell
            spike_count_df = self.build_spike_count_maps(cell_df, valid_bins)
            # 1b. build smoothed rate maps
            RM = self.build_rate_maps(spike_count_df, occupancy_map)
            if return_rate_maps:
                rate_map[:,:,i] = RM
            # 2. compute mutual information between firing and position for each cell
            spatial_info_bps[i], _ = compute_spatial_information(RM, occupancy_map)

        if return_rate_maps:
            return rate_map, spatial_info_bps
        else:
            return spatial_info_bps

    def build_occupancy_map(self, filt_vid_df, valid_pairs):
        """This function takes in a dataframe of video frames and positions, 
        and computes the occupancy (time spent) of the mouse in each spatial bin across the arena.
        Count the number of unique frame_ids (occupancy) in each bin (divide by 40 to get occupancy in seconds, since video is at 40fps)
        gaussian smooth the output matrix
        NB output are now valid bins for the place cell analysis: bins with a minimum occupancy of 1 second in this condition
        RETURNS a smoothed occupancy map (as a matrix)"""
        # count number of unique frame_ids in each xy bin pair
        occupancy_map = (filt_vid_df.filter(pl.col("xbins").is_not_nan() & pl.col("ybins").is_not_nan())  # Add this filter
                                    .with_columns(pl.col("xbins").cast(pl.Int64), pl.col("ybins").cast(pl.Int64))
                                    .group_by(["xbins", "ybins"])
                                    .agg(pl.col("frames").n_unique().alias("occupancy_frames"))
                                    .with_columns(pl.col("occupancy_frames") / self.aefizz.session.video.fps)) # convert to seconds
        # make sure that xybin pairs that had no occupancy are included in the rate map with an occupancy of 0, while bins in which the mouse was never present (outside circular arena) will be marked as NaN
        occupancy_map = (self.grid.join(valid_pairs, on=["xbins", "ybins"], how="left")
                                    .join(occupancy_map, on=["xbins", "ybins"], how="left")
                                    .with_columns(pl.when(pl.col("is_valid").is_null())
                                                    .then(None)
                                                    .otherwise(pl.col("occupancy_frames").fill_null(0))
                                                    .alias("occupancy_seconds"))
                                    .drop("is_valid", "occupancy_frames"))
        # set bins with occupancy less than the minimum threshold to NaN (these bins will be excluded from the analysis)
        occupancy_map = occupancy_map.with_columns(pl.when(pl.col("occupancy_seconds") < self.aefizz.settings.place_cell_min_occupancy)
                                                    .then(None)
                                                    .otherwise(pl.col("occupancy_seconds"))
                                                    .alias("occupancy_seconds"))
        # identify an updated set of valid bins (these are bins that had sufficient occupancy and are inside the arena)
        valid_pairs = (occupancy_map.filter(pl.col("occupancy_seconds").is_not_null())
                            .select(["xbins", "ybins"])
                            .unique()
                            .with_columns(pl.lit(True).alias("is_valid")))
        # gaussian smooth the occupancy map, ignoring nans
        smooth_occ_map = smooth_maps(occupancy_map, column="occupancy_seconds", sigma=self.aefizz.settings.place_cell_smoothing_sigma)
        return smooth_occ_map, valid_pairs
    
    def build_spike_count_maps(self, spike_count_df, valid_pairs):
        """This function takes in a dataframe of spike times and positions,
        and computes the spike count of each cell in spatial bins across the arena."""
        
        # 0. Remove rows with NaN in xbins or ybins (these are frames where the mouse's position is outside the arena, so we can't assign a bin)
        spike_count_df = spike_count_df.filter(pl.col("xbins").is_not_nan() & pl.col("ybins").is_not_nan())
        
        # 1. Count the spikes in each bin (each row in the dataframe is a frame, so we can just sum the spike counts for all the frames in each bin - NB some bins wil be empty!!)
        # bins in which the mouse was present but no spikes were recorded will have a spike count of 0, 
        # while bins in which the mouse was never present (outside circular arena) will be marked as NaN
        if (not type(spike_count_df["xbins"][0]) == int) | (not type(spike_count_df["ybins"][0]) == int):
            spike_count_df = spike_count_df.with_columns(pl.col("xbins").cast(pl.Int64),
                                               pl.col("ybins").cast(pl.Int64))
        # count number of spikes in each xy bin pair
        agg = (spike_count_df.group_by(["xbins", "ybins"])
                             .agg(pl.col("spike_count").sum().alias("spike_count")))
        # make sure that xybin pairs that had no spikes are included in the rate map with a spike count of 0
        spike_count_map = (self.grid.join(valid_pairs, on=["xbins", "ybins"], how="left")
                        .join(agg, on=["xbins", "ybins"], how="left")
                        .with_columns(pl.when(pl.col("is_valid").is_null())
                                        .then(None)
                                        .otherwise(pl.col("spike_count").fill_null(0))
                                        .alias("spike_count"))
                        .drop("is_valid"))
        # make sure "spike_count" column is float to allow for nan values
        spike_count_map = spike_count_map.with_columns(pl.col("spike_count").cast(pl.Float64))
        
        return spike_count_map

    def build_rate_maps(self, spike_count_map, occupancy_map):
        """This function takes in the spike count maps (as a dataframe) and occupancy map (as a smoother matrix) for each cell, 
        and computes the firing rate map for each cell by dividing the spike count by the occupancy in each bin."""
        
        # 1. smooth the spike map (e.g. with a Gaussian kernel) / ignoring nans! this will remove zeros before dividing
        spike_count_map = smooth_maps(spike_count_map, column="spike_count", sigma=self.aefizz.settings.place_cell_smoothing_sigma)

        # 2. compute firing rate in each bin (spike count / occupancy) - both are matrices now!
        rate_map = spike_count_map//occupancy_map

        return rate_map

    def linear_shift_place_field(self, filt_spike_df, filt_vid_df):
        """This function performs a linear shift test to determine if the mutual information is significant."""

        results = {}

        # 1. compute place field and spatial information for the FULL real data
        # filter data exclude time in shelter, escapes and when the mouse is stationary or slow (speed < 2.5 cm/s)
        filt_vid_df_explore = filter_video_dataframe(filt_vid_df, outofshelter=True, exclude_escape=True,
                                                    exclude_homings=True,
                                                    speed_threshold=self.aefizz.settings.place_cell_speed_threshold)
        filt_spike_df_explore = filt_spike_df.filter(pl.col("frames").is_in(filt_vid_df_explore["frames"]))
        # build the occupancy map (time spent in each bin) 
        results["occupancy_map"], valid_occ_pairs = self.build_occupancy_map(filt_vid_df_explore, self.valid_pairs)
        # compute place fields for each cell in this condition
        results["rate_map"], results["spatial_info_bps"] = self.compute_place_fields_cells(
            filt_spike_df_explore, results["occupancy_map"], valid_occ_pairs, return_rate_maps=True
        )

        # 2. Set up linear shift
        # select center of the filtered data and define the shifts in frames
        n_rows = len(filt_vid_df_explore)
        max_shift_one_side = (self.aefizz.settings.linshift_min_step
                            + self.aefizz.settings.linshift_step * self.aefizz.settings.linshift_step_n // 2)
        center_slice = slice(max_shift_one_side, n_rows - max_shift_one_side)

        shifts_one_side = np.arange(self.aefizz.settings.linshift_min_step, max_shift_one_side + 1,
                                    self.aefizz.settings.linshift_step)
        shifts = np.concatenate([-shifts_one_side[::-1], [0], shifts_one_side])

        # Get the arrays we need, ORDERED by the filtered video dataframe
        # These are parallel arrays: index i corresponds to the same time point
        vid_frames = filt_vid_df_explore["frames"].to_numpy()
        vid_xbins = filt_vid_df_explore["xbins"].to_numpy()
        vid_ybins = filt_vid_df_explore["ybins"].to_numpy()

        # 3. Compute real score at shift=0 (the null!) AND all shifted scores
        results["spatial_info_bps_shifted"] = np.full((len(shifts), len(self.aefizz.cluster_Ids)), np.nan)

        i = 0
        for shift in shifts:
            # POSITION comes from the center window (always the same - preserving behavioral stats)
            center_positions_xbins = vid_xbins[center_slice]
            center_positions_ybins = vid_ybins[center_slice]

            # shift the frame identities to grab spikes from a different time window
            # NB alternative approach is to add the shift to the frame number, instead of grabbing different frame numbers.
            shifted_frames = vid_frames[center_slice.start + shift : center_slice.stop + shift]

            # SPIKES come from the shifted window
            # Build a temporary video df with CENTER positions but SHIFTED frame labels
            # so that when we grab spikes by frame, we pair them with the center positions
            # NB it is important to filter and then shift the labels in the video_df and then grabbing corresponding spikes
            # so that the behavioral stats are actually preserved!
            center_vid_df = pl.DataFrame({
                "frames": shifted_frames,  # these frame numbers will match spike times from the shifted window
                "xbins": center_positions_xbins,  # but positions are from the center window
                "ybins": center_positions_ybins,
            })

            # Build occupancy from center positions (same for all shifts)
            if i == 0:
                occupancy_map_center, valid_occ_pairs_center = self.build_occupancy_map(
                    center_vid_df, self.valid_pairs
                )

            # Get spikes that fall in the shifted frames
            filt_shift_spike_df = filt_spike_df.filter(pl.col("frames").is_in(shifted_frames))

            # CRITICAL: overwrite the spike positions with the CENTER positions
            # Create a mapping: shifted_frame -> center_position
            filt_shift_spike_df = (filt_shift_spike_df
                                .select(["frames", "spike_clusters","spike_count"])
                                .join(center_vid_df, on="frames", how="left"))

            # Now compute place fields — spikes from shifted time, positions from center time
            if shift == 0:
                # For the real score at shift=0, we want to return the rate maps for plotting later, so set return_rate_maps=True
                results["rate_map_null"], results["spatial_info_bps_null"] = self.compute_place_fields_cells(
                    filt_shift_spike_df, occupancy_map_center, valid_occ_pairs_center, return_rate_maps=True
                )
            else:
                results["spatial_info_bps_shifted"][i] = self.compute_place_fields_cells(
                    filt_shift_spike_df, occupancy_map_center, valid_occ_pairs_center, return_rate_maps=False
                )
                i += 1

        return results
    
    def plot_place_fields_conditions(self):
        logger.info("Plotting place fields for each condition and cluster")
        plot_folder = make_directory(os.path.join(self.savepath, "PC_plots_" + self.hexaname))
        for idx, Id in enumerate(self.aefizz.cluster_Ids):
            # find the min and max across both the rate map and null rate map across all conditions for this cluster to set the same color scale for all plots
            min_rate = np.min([np.nanmin(self.results_dict[c]["rate_map"][:,:,idx]) for c in self.aefizz.all_conditions] + 
                              [np.nanmin(self.results_dict[c]["rate_map_null"][:,:,idx]) for c in self.aefizz.all_conditions])
            max_rate = np.max([np.nanmax(self.results_dict[c]["rate_map"][:,:,idx]) for c in self.aefizz.all_conditions] + 
                              [np.nanmax(self.results_dict[c]["rate_map_null"][:,:,idx]) for c in self.aefizz.all_conditions])
            fig, axs = plt.subplots(3, len(self.aefizz.all_conditions), figsize=(5*len(self.aefizz.all_conditions), 3*5))
            for j, c in enumerate(self.aefizz.all_conditions):
                # plot rate map for real data
                Arena(ax = axs[0,j], dim = self.results_dict[c]["rate_map"][:,:,idx].shape[0]-1, condition = c,
                      barrier_coordinates = self.aefizz.session.barrier_location[:-1], full_image = False)
                im = axs[0,j].imshow(self.results_dict[c]["rate_map"][:,:,idx], vmin=min_rate, vmax=max_rate)
                axs[0,j].set_title(f"Real data - {c}")
                # plot rate map for null data
                Arena(ax = axs[1,j], dim = self.results_dict[c]["rate_map_null"][:,:,idx].shape[0]-1, condition = c,
                      barrier_coordinates = self.aefizz.session.barrier_location[:-1], full_image = False)
                im = axs[1,j].imshow(self.results_dict[c]["rate_map_null"][:,:,idx], vmin=min_rate, vmax=max_rate)
                axs[1,j].set_title(f"Null data")
                # plot spatial information for shifted data
                axs[2,j].hist(self.results_dict[c]["spatial_info_bps_shifted"][:,idx], bins=20)
                axs[2,j].axvline(self.results_dict[c]["spatial_info_bps"][idx], color="red", label="Real data SI")
                axs[2,j].axvline(self.results_dict[c]["spatial_info_bps_null"][idx], color="black", label="Null data SI")
                axs[2,j].set_title(f"Spatial info (bps): {self.results_dict[c]['spatial_info_bps'][idx]:.2f}")
                axs[2,j].legend()
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="Firing rate")
            plt.savefig(os.path.join(plot_folder, f"place_fields_cluster{str(Id)}.png"))
            plt.close()

    def save(self):
        """This function saves the results of the place cell analysis to a file."""
        logger.info("Saving place cell results to file and database")
        filename = os.path.join(self.savepath, "PC_" + self.hexaname)
        np.savez(os.path.join(filename + "_results.npz"), 
                             **self.results_dict,
                             allow_pickle=True)
        settings=asdict(self.aefizz.settings)
        np.savez(filename + "_settings.npz", **settings, allow_pickle=True)
        # add results to database
        add_run_to_database(self.database, 
                            settings_to_check(self.aefizz.settings,["linshift", "place_cell"]), 
                            self.savepath + os.sep + "place_cell_results.csv", 
                            self.hexaname)

    # --------------Helper Functions--------------
    def check_database_for_same_run(self):

        self.do_analysis = True
        # check if database file exists
        if os.path.exists(self.savepath + os.sep + "place_cell_results.csv"):
            self.database = pd.read_csv(self.savepath + os.sep + "place_cell_results.csv")
            # check if there is a run with the same settings as the current ones
            matched_results = check_database_for_matched_results(self.database, settings_to_check(self.aefizz.settings,["linshift", "place_cell"]))
            if len(matched_results) > 0:
                # if there is a match, print the name of the matched run and skip the analysis....
                logger.info(f"Found {len(matched_results)} matched results in database for current settings: {matched_results} in the folder {self.savepath}")
                self.do_analysis = False
                if self.aefizz.settings.redo_compute:
                    #... unless you have chosen to redo the analysis anyway!
                    logger.info("You have chosen to redo the analysis anyway!")
                    self.do_analysis = True
        else:
            # if database doesn't exist, create it and add the current run to the database
            logger.info(f"No existing database found for place cell results at {self.savepath}, will compute place cell results and save to new database.")
            self.database = pd.DataFrame([])
        
        # if we are doing the place cell analysis, add the current run to the database with a new hexadecimal name
        if self.do_analysis:
            self.hexaname = generate_run_id()