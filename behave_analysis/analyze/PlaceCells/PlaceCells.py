"""A script for computing place cells!"""

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.analyze.PlaceCells.place_cell_utils import assign_positional_bins_to_frames

class PlaceCells:
    def __init__(self, aefizz):
        self.aefizz = aefizz

    def compute_place_cells(self):
        """This function identifies place cells based on their firing rate maps and mutual information with position.
        It iterates through each cell in each condition."""

        # assign each position to a bin
        bin_size = 5  # in cm, adjust as needed
        arena_size = (self.aefizz.session.video.radius*2)/self.aefizz.session.video.pixels_per_cm
        nbins = int(arena_size / bin_size)  # calculate number of bins based on arena size and desired bin size
        df = assign_positional_bins_to_frames(self.aefizz.video_and_spike_data, nbins=nbins)

        for c in self.aefizz.conditions:
            # 0. filter time: exclude time in shelter, escapes and when the mouse is stationary or slow (speed < 2.5 cm/s)
            filtered_df = filter_video_dataframe(self.aefizz.video_and_spike_data, c, outofshelter=True, exclude_escape=True, speed_threshold=self.aefizz.settings.min_speed_threshold)
        
            for n in self.aefizz.cluster_Ids:
                df = [] # get spikes only for this cell
                # 1. build rate maps for each cell
                self.build_rate_maps(df, c)
                # 2. compute mutual information between firing and position for each cell
                self.compute_mutual_information()
                # 3. perform linear shift test to determine significance of mutual information
                self.linear_shift_significance()

    def build_rate_maps(self, dataframe, condition):
        """This function takes in a dataframe of spike times and positions,
        and computes the firing rate of each cell in spatial bins across the arena."""
        
        # 1. bin the arena into spatial bins (e.g. 5cm x 5cm)
        # 2. for each cell, compute the firing rate in each spatial bin
        # 3. check for bins with sufficient occupancy (e.g. at least 1s)
        # 4. smooth the rate maps (e.g. with a Gaussian kernel)
        pass

    def compute_mutual_information(self):
        """This function computes the mutual information between the firing of each cell and the mouse's position."""
        pass

    def linear_shift_significance(self):
        """This function performs a linear shift test to determine if the mutual information is significant."""
        pass
