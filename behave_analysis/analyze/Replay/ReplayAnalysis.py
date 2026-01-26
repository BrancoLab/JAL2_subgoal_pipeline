"""This is a class for replay analysis.
It will identify putative reactivation events and then check if they replay sequences (compare to template)

Define period of interest:
1. time when a fraction of cells are active
2. time before homing or when mouse is in shelter after escape

Compare activity during these periods to templates:
1. Rank-order correlation (based on Diba & Buzsaki, 2007)
2. Bayesian decoding (based on Davidson, Kloosterman and Wilson 2009; replay score or radon transform or linear weighted correlation)
3. State space decoder (based on Denovellis, ..., Frank, 2021; https://github.com/Eden-Kramer-Lab/replay_trajectory_classification)

TODO: how to compare tuned and untuned neurons?
TODO: how to handle separate conditions?!
"""

import os
import numpy as np
import dill as pickle
from scipy.stats import zscore, spearmanr
import warnings

from behave_analysis.analyze.Replay.load_cells import load_hdir_cells, load_escape_tuned_cells
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import load_or_compute_escape_tuning
from behave_analysis.analyze.Replay.BayesianDecoderFunctions import (bayesian_decoder,
                                                                     calculate_linear_weighted_correlation,
                                                                     calculate_custom_replay_score,
                                                                     calculate_radon_score)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.Replay.Replay import Replay
from behave_analysis.analyze.Replay.StateSpaceDecoderDataFormatter import prepare_state_space_decoder_data
from replay_trajectory_classification import SortedSpikesDecoder, Environment, RandomWalk, estimate_movement_var

class ReplayAnalysis:

    def __init__(self, aefizz):
        self.aefizz = aefizz
        self.replay = Replay(settings = self.aefizz.settings)
        self.replay.savepath = make_directory(os.path.join(self.aefizz.session.base_path, 
                                                            self.aefizz.session.processed_path, 
                                                            "replay", 
                                                            self.aefizz.settings.replay_time_period,))

    def select_cells_of_interest(self):
        """Select cells to include in replay analysis based on settings. 
        These are the cells for which we will look at reactivation events."""

        if self.aefizz.settings.replay_cells == "hdir":
            load_hdir_cells()
        elif self.aefizz.settings.replay_cells == "escape_tuned":
            # boolean array of shape (num_cells, 3) indicating significant escape tuned cells for each condition
            xval =load_escape_tuned_cells(self.aefizz) 
            self.replay.selected_cells = np.where(np.any(xval, axis=1))[0]
        elif self.aefizz.settings.replay_cells == "all":
            self.replay.selected_cells = np.full(self.aefizz.frame_by_cluster_matrix.shape[1], True)

        self.fcm = self.aefizz.frame_by_cluster_matrix[:, self.replay.selected_cells]

        pass
    
    def define_replay_template(self):
        """The template is the order of neurons during a sequence that we want to test for replay"""
        var = "escape"
        time_period = "homing&escape"

        # load in escape homing/escape tuning curve
        CT = load_or_compute_escape_tuning(self.aefizz, self.aefizz.settings.escape_tuning_bins, var, time_period)
        self.escape_tuning_curve = CT.fr_full[c, self.selected_cells,]  # tuning curves of selected cells
        # define the template of the order of neurons in the sequence
        preferred_tuning = CT.params_full[:,:,1] # preferred (max) bin for each cell and condition
        preferred_tuning = preferred_tuning[self.selected_cells,:]  # only selected cells
        self.replay.template_seq = np.full_like(preferred_tuning, np.nan)
        for c in range(preferred_tuning.shape[1]):
            self.replay.template_seq[:,c] = np.argsort(preferred_tuning[:,c])
        # load in discretized escape positions for occupancy prior
        self.discretized_var = CT.discretized_var[CT.conditions == c]

    def filter_time(self):
        """Filter time periods to include in replay analysis.
        E.g., only time before homing or time in shelter after escape.
        RETURNS: self.time_mask: boolean array of shape (num_timepoints,) indicating timepoints to include
        """
        if self.aefizz.settings.replay_time_period == "before_homing":
            # look at the 2s before homing onset
            # TODO: this could be made to only use subsets of homings (e.g. only long homings) 
            # by using a curated list instead of homing_object.onset_frames
            homing_onset_bool = np.full(self.aefizz.video_df.shape[0], False)
            homing_onset_bool[self.aefizz.homings_object.onset_frames] = True
            window = np.concatenate((np.full((2*self.aefizz.session.video.fps,),True), np.full((2*self.aefizz.session.video.fps,),False)))  # 2s window at 40Hz
            self.replay.time_mask = (np.convolve(homing_onset_bool.astype(int), window.astype(int), mode='same') > 0)

        elif self.aefizz.settings.replay_time_period == "in_shelter_after_escape":
            # look at the 2s after shelter entry following an escape
            shelter_entry_after_escape = np.where((self.aefizz.video_df["EscapePeriod"] == True) & 
                                                  (self.aefizz.video_df['OutofshelterIdx'].to_numpy() == False))[0]
            shelter_entry_vec = np.full(self.aefizz.video_df.shape[0], False)
            shelter_entry_vec[shelter_entry_after_escape] = True
            window = np.concatenate((np.full((2*self.aefizz.session.video.fps,),False), np.full((2*self.aefizz.session.video.fps,),True)))  # 2s window at 40Hz
            self.replay.time_mask = ((np.convolve(shelter_entry_vec.astype(int), window.astype(int), mode='same') > 0) & 
                             (self.aefizz.video_df['OutofshelterIdx'].to_numpy() == False))
            
        elif self.aefizz.settings.replay_time_period == "outside_shelter":
            # any time the mouse is outside the shelter
            self.replay.time_mask = self.aefizz.video_df['OutofshelterIdx'].to_numpy() == True

        elif self.aefizz.settings.replay_time_period == "stationary_outside_shelter":
            # any time the mouse is outside the shelter and stationary
            self.replay.time_mask = ((self.aefizz.video_df['OutofshelterIdx'].to_numpy() == True) &
                              (self.aefizz.video_df['speed'].to_numpy() < 0.5))

        elif self.aefizz.settings.replay_time_period == "in_shelter":
            # any time the mouse is inside the shelter
            self.replay.time_mask = self.aefizz.video_df['OutofshelterIdx'].to_numpy() == False

    def identify_time_windows_of_interest(self):
        """Identify periods of time where a fraction of cells are active
        based on a zscore threshold within a sliding window.
        This gives us windows of time with putative reactivation events to test for replay.
        RETURNS: self.time_mask: boolean array of shape (num_timepoints,) indicating timepoints of the identified windows
        """
        window_samples = int((self.aefizz.settings.replay_search_window/1000)*40)
        step = int(((self.aefizz.settings.replay_search_window/1000)/3)*40) # step every 100ms
        threshold = 2 # zscore threshold
        frac_active = .3

        identified_window = []
        identified_num_active = []
        total_time = self.fcm.shape[0]
        activity = np.full(total_time, False)
        fcm_z = zscore(self.fcm, axis=0)

        # loop through periods where bool_mask is true to find active windows
        b_start = np.where(np.diff(self.time_mask.astype(int)) == 1)[0]
        b_end = np.where(np.diff(self.time_mask.astype(int)) == -1)[0]
        for s,e in zip(b_start, b_end):
            if e-s < window_samples:
                continue
            # check if there is a window of activity
            for i in range(s, e-window_samples, step):
                window = fcm_z[i:i+window_samples,:] > threshold
                time_over_t = np.sum(window, axis = 0) # number of timepoints above threshold
                num_active = np.sum(time_over_t > 0) # number of active cells
                # check if 30% of cells are active
                if num_active > (frac_active * window.shape[1]):
                    activity[i:i+window_samples] = True
                    identified_window.append(i)
                    identified_num_active.append(num_active)

        # there will be some consecutive windows with overlapping activity
        # find consecutive windows, look at which one has the highest fraction of active cells
        # only keep the one with the most active cells

        # Step 1: Find breaks between consecutive windows
        diff_window_time = np.diff(identified_window)
        breaks = np.where(diff_window_time > 40)[0]
        breaks = np.append(breaks, len(identified_window)-1)  # Add the last index

        # Step 2: Process each group of consecutive windows
        start_idx = 0
        best_activity = np.full(total_time, False)  # Initialize best activity array

        for end_idx in breaks:
            # Extract the current group of consecutive windows
            group = identified_window[start_idx:end_idx+1]
            
            if len(group) == 0:
                continue
                
            # Find the window with highest fraction of active cells in this group
            group_fractions = [identified_num_active[a] for win_idx in group for a,x, in enumerate(identified_window) if x == win_idx]
            best_window_idx = start_idx + np.argmax(group_fractions)

            # Update the best activity array
            best_activity[identified_window[best_window_idx]:identified_window[best_window_idx] + window_samples] = True
            
            # Move to the next group
            start_idx = end_idx + 1

        self.replay.time_mask = best_activity

    def rank_order_correlation(self):
        """Compute rank-order correlation between activity during identified windows and template sequence.
        Based on Diba & Buzsaki, 2007.
        TODO: ugh, shouldn't correlation be measured between event_seq and template_seq?
        RETURNS: correlation_pass: boolean array of shape (num_timepoints,) indicating windows that pass correlation"""

        correlation_threshold = 0.2
        zscore_threshold = 2
        correlation_pass = np.full(self.fcm.shape[0], False) # boolean array of shape (num_timepoints,) indicating windows that pass correlation

        window_onsets = np.where(np.diff(self.time_mask.astype(int)) == 1)[0]
        window_offsets = np.where(np.diff(self.time_mask.astype(int)) == -1)[0]

        for onset, offset in zip(window_onsets, window_offsets):
            # threshold the activity in the window
            n_window = zscore(self.fcm[onset:offset,:], axis=0)

            # event_seq based on first activity as moment when it crosses threshold
            if self.aefizz.settings.replay_rank_order_corr_method == 'first_activity':
                first_activity = np.zeros(n_window.shape[1])
                for n in range(n_window.shape[1]):
                    if len(np.where((n_window[:,n] > zscore_threshold) == True)[0]) > 0:
                        first_activity[n] = np.where((n_window[:,n] > zscore_threshold) == True)[0][0]
                event_seq = np.argsort(first_activity)

            # event seq based on weighted average of activity to find where the bump is
            if self.aefizz.settings.replay_rank_order_corr_method == 'weighted_avg':
                weighted_avg = np.zeros(n_window.shape[1])
                for n in range(n_window.shape[1]):
                    weighted_avg[n] = np.sum(n_window[:,n] * np.arange(1, (offset-onset)+1)) / np.sum(n_window[:,n])
                event_seq = np.argsort(weighted_avg)
            
            # rank order correlation
            correlation, _ = spearmanr(np.arange(n_window.shape[1]), event_seq)
            if np.abs(correlation) > correlation_threshold:
                correlation_pass[onset:offset] = True

    def bayesian_decoder(self):
        """Compute Bayesian decoding of activity during identified windows and compare to template sequence.
        Based on Davidson, Kloosterman and Wilson 2009; replay score or radon transform or linear weighted correlation.
        INPUTS:
            firing_rate_maps: A 2D array or list of arrays. Shape: (n_neurons, n_position_bins). 
                            Contains the average firing rate of each neuron i for each position bin x, sorted by template_seq. These are the "templates".
            spike_counts: A 2D array. Shape: (n_time_bins, n_neurons). 
                        Contains the number of spikes n_i detected for each neuron i within each time bin t of the candidate event.
            occupancy_map: A 1D array. Shape: (n_position_bins,). 
                        Contains the normalized probability P(x) of the animal being in each position bin x, derived from overall session behavior.
            time_bin_width (τ): A float. The duration of each time bin in seconds (e.g., 0.02 for 20ms).
            n_neurons (N): An integer. The total number of neurons used for decoding.
            n_position_bins: An integer. The number of spatial bins used to discretize the environment.
            n_time_bins: An integer. The number of time bins in the candidate event window.
        """

        max_fract = 100
        fract_thresh = 10.0 # Threshold of fraction of route in percent
        # Define parameter search ranges (adjust based on expected replay speeds/positions)
        # Velocity range (e.g., -10 m/s to +10 m/s, 101 steps) -> -1000 cm/s to 1000 cm/s
        V_range = [-50, 50, 51]
        # Starting position range (e.g., full track length, 100 steps)
        rho_range = [0, max_fract, 100]
        time_bin_width = 1/self.aefizz.session.video.fps  # 25 ms for a 40Hz frame rate

        firing_rate_map = self.escape_tuning_curve[self.template_seq,:]  # shape (n_neurons, n_position_bins)
        sorted_fcm = self.fcm[:, self.template_seq.flatten()]

        # set up some variables
        n_position_bins = firing_rate_map.shape[1]
        
        # Define position bins (replace with your actual bins)
        position_bin_edges = np.linspace(0, max_fract, n_position_bins + 1)

        # create the prior
        if self.aefizz.settings.occupancy_prior == 'uniform':
            occupancy_map = np.ones(n_position_bins) / n_position_bins # Uniform prior
        elif self.aefizz.settings.occupancy_prior == 'empirical':
            occupancy_counts = np.bincount(self.discretized_var.astype(int), minlength=n_position_bins)
            occupancy_map = occupancy_counts / np.sum(occupancy_counts)

        onsets = np.where(np.diff(self.time_mask.astype(int)) == 1)[0]
        offsets = np.where(np.diff(self.time_mask.astype(int)) == -1)[0]

        for onset, offset in zip(onsets, offsets):
            spike_counts = sorted_fcm[onset:offset,:]
            n_time_bins = spike_counts.shape[0]

        self.replay.bayesian_posterior = bayesian_decoder(firing_rate_map, spike_counts, occupancy_map, time_bin_width, n_time_bins, n_position_bins)
        self.replay.radon_score, self.replay.radon_angle = calculate_radon_score(self.replay.bayesian_posterior)
        self.replay.linear_corr = calculate_linear_weighted_correlation(self.replay.bayesian_posterior)
        self.replay.R_max, self.replay.V_max, self.replay.rho_max, self.replay.R_map = calculate_custom_replay_score(self.replay.bayesian_posterior, time_bin_width, position_bin_edges, fract_thresh, V_range, rho_range)

    def state_space_decoder(self):
        """Compute state space decoding of activity during identified windows and compare to template sequence.
        Based on Denovellis, ..., Frank, 2021
        INPUTS:
            position
            spikes
            time
        RETURNS:
            causal_posterior: the probability of position given only past spikes
            acausal_posterior: the probability of position given all past and future spikes
        TODO:It also requires loading their package 'replay_trajectory_classification'
            How to divide train and test data?"""   

        np.warnings = warnings
        # not implemented yet
        # assert False, "State space decoder not implemented yet"

        # prepare data

        filename = self.replay.savepath + os.sep + self.aefizz.settings.replay_template_match_method + "_state_space_decoder_data_bin" + str(self.aefizz.settings.replay_state_space_decoder_bin_size) + ".npz"
        if os.path.exists(filename):
            data = np.load(filename)
            spikes = data['spikes']
            time = data['time']
            position = data['position']
        else:
            spikes, time = prepare_state_space_decoder_data(self.aefizz.spike_df, self.replay.time_mask, self.aefizz.settings.replay_state_space_decoder_bin_size)
            var = 
            np.savez(filename, spikes=spikes, time=time, position=position)

        # set up environment parameters
        linear_track_env = Environment(
                                    place_bin_size=1,  # Adjust based on your actual bin size
                                    # track_graph=my_linear_graph,  # Define a linear graph with 25 nodes
                                    edge_order=[(i, i + 1) for i in range(24)],  # Sequential edges
                                    edge_spacing=None,  # No gaps between edges
                                    position_range=[(0, 25)],  # 0 to 100% of escape route
                                    infer_track_interior=False,  # Data is already binned
                                    fill_holes=False,
                                    dilate=False,
                                    bin_count_threshold=1,
                                )
        
        transition_type = RandomWalk(movement_var=1)

        decoder = SortedSpikesDecoder(
                                    environment=linear_track_env,
                                    transition_type=transition_type,
                                    sorted_spikes_algorithm='spiking_likelihood_kde',
                                    sorted_spikes_algorithm_params={'block_size': None,
                                                                    'position_std': [1.0],
                                                                    'use_diffusion': False},
                                )

        decoder.fit(position[:int(spikes.shape[0]/2)], spikes[:int(spikes.shape[0]/2),:])

        results = decoder.predict(spikes[int(spikes.shape[0]/2):,:], time=time[int(spikes.shape[0]/2):])

        self.replay.state_space_causal_posterior = results.causal_posterior
        self.replay.state_space_acausal_posterior = results.acausal_posterior

    def save_outputs(self):
        """Save outputs of replay analysis to self.aefizz.replay_analysis_results"""
        filename = self.replay.savepath + os.sep + self.aefizz.settings.replay_template_match_method + "_" + self.aefizz.settings.replay_cells + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.replay, f)