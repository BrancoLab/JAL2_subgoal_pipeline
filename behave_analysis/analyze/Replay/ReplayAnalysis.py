"""This is a class for replay analysis.
It will identify putative reactivation events and then check if they replay sequences (compare to template)

Define period of interest:
1. time when a fraction of cells are active
2. time before homing or when mouse is in shelter after escape

Compare activity during these periods to templates:
1. Rank-order correlation (based on Diba & Buzsaki, 2007)
2. Bayesian decoding (based on Davidson, Kloosterman and Wilson 2009; replay score or radon transform or linear weighted correlation)
3. State space decoder (based on Denovellis, ..., Frank, 2021; https://github.com/Eden-Kramer-Lab/replay_trajectory_classification)

TODO: how to compare tuned and untuned neurons? in pre homie! look at fract of active cells!
TODO: how to handle separate conditions?!
      How to divide train and test data? train on good homies, test on bad ones. or train on good homies, tes on pre homie period
"""

import os
import numpy as np
from scipy.stats import zscore, spearmanr
from loguru import logger
from dataclasses import asdict


from behave_analysis.analyze.Replay.load_cells import load_hdir_cells, load_escape_tuned_cells
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import homing_escape_onsets, create_discretized_behave_var
from behave_analysis.analyze.EscapePattern.ComputeEscapeTuning import load_or_compute_escape_tuning
from behave_analysis.analyze.Replay.BayesianDecoderFunctions import (bayesian_decoder,
                                                                     calculate_linear_weighted_correlation,
                                                                     calculate_custom_replay_score,
                                                                     calculate_radon_score)
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.Replay.Replay import Replay
from behave_analysis.analyze.Replay.StateSpaceDecoderDataFormatter import prepare_state_space_decoder_data

class ReplayAnalysis:

    def __init__(self, aefizz):
        self.aefizz = aefizz
        self.replay = Replay(settings = self.aefizz.settings)
        self.check_settings_compatibility()
        self.replay.savepath = make_directory(os.path.join(self.aefizz.session.base_path, 
                                                            self.aefizz.session.processed_path, 
                                                            "models",
                                                            "replay",
                                                            "replay_" + self.aefizz.settings.replay_template_match_method, 
                                                            self.aefizz.settings.replay_decoder_test_time_period + "_" + self.aefizz.settings.replay_condition))
        self.c = [x for x, c in enumerate(["shelter_only", "barrier_pre_flip", "barrier_post_flip"]) if c == self.aefizz.settings.replay_condition][0]

    def find_replay_rank_order_correlation(self):
        """Main function to find replay events using rank-order correlation method."""
        logger.warning("This has not been debugged! We're not currently saving the data anywhere.")
        self.select_cells_of_interest()
        self.define_replay_template()
        self.replay.test_time_mask = self.filter_time(self.aefizz.settings.replay_decoder_test_time_period)
        putative_replay_window_bool = self.identify_time_windows_of_interest(self.replay.test_time_mask)
        self.rank_order_correlation(putative_replay_window_bool)

    def find_replay_bayesian_decoder(self):
        """Main function to find replay events using Bayesian decoder method."""
        logger.warning("This has not been debugged! Train and test periods need to be split! Results need to saved.")
        self.select_cells_of_interest()
        self.replay.test_time_mask = self.filter_time(self.aefizz.settings.replay_decoder_test_time_period)
        test_behave, _ = self.prepare_behavioral_variable(self.replay.test_time_mask, tuning_var = 'escape')
        self.bayesian_decoder(self.replay.test_time_mask, test_behave)

    def find_replay_state_space_decoder(self):
        """Main function to find replay events using State Space decoder method."""
        self.select_cells_of_interest()
        self.replay.test_time_mask = self.filter_time(self.aefizz.settings.replay_decoder_test_time_period)
        self.replay.train_time_mask = self.filter_time(self.aefizz.settings.replay_decoder_train_time_period)
        train_behave, train_condition = self.prepare_behavioral_variable(self.replay.train_time_mask, tuning_var = 'escape') # we assume train_time_period is always homing&escape
        if self.aefizz.settings.replay_decoder_test_time_period in ["homing", "escape", "homing&escape"]:
            # because we can only compute %escape on homing&escape periods
            test_behave, test_condition = self.prepare_behavioral_variable(self.replay.test_time_mask, tuning_var = 'escape')
        else:
            test_behave, test_condition = self.prepare_behavioral_variable(self.replay.test_time_mask)
        # filter by condition 
        self.replay.train_time_mask[np.where(self.replay.train_time_mask)[0][train_condition != self.c]] = False
        self.replay.test_time_mask[np.where(self.replay.test_time_mask)[0][test_condition != self.c]] = False
        self.state_space_decoder(self.replay.train_time_mask, train_behave, self.replay.test_time_mask, test_behave)

# ------------ Replay functions ------------

    def select_cells_of_interest(self):
        """Select cells to include in replay analysis based on settings. 
        These are the cells for which we will look at reactivation events."""

        hdir = load_hdir_cells(self.aefizz.session)
        if self.aefizz.settings.replay_cells == "escape_tuned":
            # boolean array of shape (num_cells, 3) indicating significant escape tuned cells for each condition
            xval = load_escape_tuned_cells(self.aefizz) 
            xval[hdir, :] = np.full((len(hdir), xval.shape[1]), False)  # remove hdir cells
            self.replay.selected_cells = np.where(xval[:, self.c])[0]
            # self.replay.selected_cells = np.where(np.any(xval, axis=1))[0] # tuned in any condition

        elif self.aefizz.settings.replay_cells == 'escape_untuned':
            xval = load_escape_tuned_cells(self.aefizz) 
            self.replay.selected_cells = np.where(~np.any(xval, axis=1))[0]

        elif self.aefizz.settings.replay_cells == "all":
            self.replay.selected_cells = np.full(self.aefizz.frame_by_cluster_matrix.shape[1], True)

        self.fcm = self.aefizz.frame_by_cluster_matrix[:, self.replay.selected_cells]

    def check_settings_compatibility(self):
        """Check that the settings for replay analysis are compatible.
        E.g., if searching for escape pattern replay, don't train on exploration periods."""
        assert self.aefizz.settings.replay_decoder_variable in ["escape", "shelter_dist"], "replay_decoder_variable must be 'escape' or 'shelter_dist'"
        assert self.aefizz.settings.replay_decoder_variable == "escape", "Currently only 'escape' variable is implemented for replay analysis"
        # now, only homing&escape, but should work for just homing, just escape without edits
        assert self.aefizz.settings.replay_decoder_train_time_period == "homing&escape", "Currently only 'homing&escape' training period is implemented for replay analysis"
        assert (self.aefizz.settings.replay_decoder_variable == "escape" and 
                self.aefizz.settings.replay_decoder_train_time_period == "homing&escape"), "If replay_decoder_variable is 'escape', replay_decoder_train_time_period must be 'homing&escape'"

    def define_replay_template(self):
        """The template is the order of neurons during a sequence that we want to test for replay"""
        # load in escape homing/escape tuning curve
        CT = load_or_compute_escape_tuning(self.aefizz, self.aefizz.settings.replay_decoder_variable + ' in ' + self.aefizz.settings.replay_decoder_train_time_period)
        self.escape_tuning_curve = CT.fr_full[self.c, self.replay.selected_cells,:]  # tuning curves of selected cells
        # define the template of the order of neurons in the sequence
        preferred_tuning = CT.params_full[:,:,1] # preferred (max) bin for each cell and condition
        preferred_tuning = preferred_tuning[self.replay.selected_cells,:]  # only selected cells
        self.replay.template_seq = np.argsort(preferred_tuning[:,self.c]) # only the condition of interest
        
    def prepare_behavioral_variable(self, time_mask, tuning_var = ''):
        """Compute behavioral variable for:
        1. occupancy prior of bayesian decoder
        2. discretized variable for state space decoder"""
        x = self.aefizz.video_df['mouse_x_position'].to_numpy()[time_mask]
        y = self.aefizz.video_df['mouse_y_position'].to_numpy()[time_mask]
        condition = np.zeros(len(self.aefizz.video_df))
        condition[self.aefizz.video_df["barrier_present"].to_numpy() == True] += 1
        condition[self.aefizz.video_df["barrier_flipped"].to_numpy() == True] += 1
        condition = condition[time_mask]
        if tuning_var == '':
            discretized_var = np.zeros(len(x))
        else:
            discretized_var = create_discretized_behave_var(self.aefizz, x, y, condition, 
                                                    tuning_var = tuning_var, 
                                                    homing_vector = time_mask, 
                                                    settings = self.aefizz.settings)
        return discretized_var, condition

    def filter_time(self, time_period):
        """Filter time periods to include in replay analysis. 
        The time period could either be the replay_decoder_train_time_period or replay_decoder_test_time_period.
        E.g., only time before homing or time in shelter after escape.
        RETURNS: self.time_mask: boolean array of shape (num_timepoints,) indicating timepoints to include
        """
        window_length = 2 # seconds
        speed_threshold = 0.5  # cm/s

        if "homing" in time_period or "escape" in time_period:
            ons, offs, _ = homing_escape_onsets(self.aefizz, time_period)
            durations = offs - ons # durations in frames

        if time_period == "before_homing":
            # look at the 2s before homing onset
            # TODO: this could be made to only use subsets of homings (e.g. only long homings) by using the homing_escape_onsets function
            homing_onset_bool = np.full(self.aefizz.video_df.shape[0], False)
            homing_onset_bool[ons.astype(int)] = True
            window = np.concatenate((np.full((window_length*self.aefizz.session.video.fps,),True), np.full((window_length*self.aefizz.session.video.fps,),False)))  # 2s window at 40Hz
            time_mask = (np.convolve(homing_onset_bool.astype(int), window.astype(int), mode='same') > 0)

        elif "in_shelter_after" in time_period:
            logger.warning("We only consider runs that are followed by shelter entry within 20s of onset or within stimulus duration.")
            # find shelter entries after escape onset
            shelter_entries = np.where(np.diff(self.aefizz.video_df['OutofshelterIdx'].to_numpy().astype(int)) == -1)[0] + 1  # +1 to get the entry frame
            on = []
            for i, d in zip(ons, durations):
                entry_after_escape = shelter_entries[shelter_entries > int(i)]
                if len(entry_after_escape) == 0:
                    continue
                if (entry_after_escape[0] - int(i)) < np.amax([20 * self.aefizz.session.video.fps, int(d)]):  # only consider shelter entries within 20s of escape onset or within stimulus duration
                    on.append(entry_after_escape[0])
            # look at the 2s after shelter entry following an escape
            shelter_entry_vec = np.full(self.aefizz.video_df.shape[0], False)
            shelter_entry_vec[on] = True
            window = np.concatenate((np.full((window_length*self.aefizz.session.video.fps,),False), np.full((window_length*self.aefizz.session.video.fps,),True)))  # 2s window at 40Hz
            time_mask = ((np.convolve(shelter_entry_vec.astype(int), window.astype(int), mode='same') > 0) & 
                             (self.aefizz.video_df['OutofshelterIdx'].to_numpy() == False)) # if mouse leaves again before the 2s are up, don't include that time
            
        elif time_period == "outside_shelter":
            # any time the mouse is outside the shelter
            time_mask = self.aefizz.video_df['OutofshelterIdx'].to_numpy() == True

        elif time_period == "stationary_outside_shelter":
            # any time the mouse is outside the shelter and stationary
            logger.warning("This gives us any single frame the mouse is stationary, not just prolonged periods of being stationary.")
            time_mask = ((self.aefizz.video_df['OutofshelterIdx'].to_numpy() == True) &
                              (self.aefizz.video_df['speed'].to_numpy() < speed_threshold))

        elif time_period == "in_shelter":
            # any time the mouse is inside the shelter
            time_mask = self.aefizz.video_df['OutofshelterIdx'].to_numpy() == False

        elif time_period == "homing&escape":
            # find shelter entries after escape onset
            shelter_entries = np.where(np.diff(self.aefizz.video_df['OutofshelterIdx'].to_numpy().astype(int)) == -1)[0] + 1  # +1 to get the entry frame
            # if mouse enters shelter before offset of homing/escape, set offset to shelter entry
            time_mask = np.full(self.aefizz.video_df.shape[0], False)
            for on, off in zip(ons, offs):
                entry_after_escape = shelter_entries[shelter_entries > int(on)]
                if len(entry_after_escape) == 0:
                    time_mask[int(on):int(off)] = True
                    continue
                if entry_after_escape[0] < off:  # only consider shelter entries within 20s of escape onset or within stimulus duration
                    off = entry_after_escape[0]
                time_mask[int(on):int(off)] = True
        
        return time_mask

    def identify_time_windows_of_interest(self, time_mask):
        """Identify periods of time where a fraction of cells are active
        based on a zscore threshold within a sliding window.
        This gives us windows of time with putative reactivation events to test for replay.
        INPUTS:
            self.replay.time_mask: boolean array of shape (num_timepoints,) indicating time periods of interest (e.g. before homing)
        RETURNS: 
           self.replay.time_mask: boolean array of shape (num_timepoints,) indicating timepoints of the identified windows
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

        # loop through periods where time_mask is true to find active windows
        b_start = np.where(np.diff(time_mask.astype(int)) == 1)[0]
        b_end = np.where(np.diff(time_mask.astype(int)) == -1)[0]
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

        return best_activity

    def rank_order_correlation(self, putative_replay):
        """Compute rank-order correlation between activity during identified windows and template sequence.
        Based on Diba & Buzsaki, 2007.
        RETURNS: 
            correlation_pass: boolean array of shape (num_timepoints,) indicating windows that pass correlation"""

        correlation_threshold = 0.2
        zscore_threshold = 2
        correlation_pass = np.full(self.fcm.shape[0], False) # boolean array of shape (num_timepoints,) indicating windows that pass correlation

        window_onsets = np.where(np.diff(putative_replay.astype(int)) == 1)[0]
        window_offsets = np.where(np.diff(putative_replay.astype(int)) == -1)[0]

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
            correlation, _ = spearmanr(self.replay.template_seq[:,self.c], event_seq)
            if np.abs(correlation) > correlation_threshold:
                correlation_pass[onset:offset] = True

    def bayesian_decoder(self, time_mask, behave_var):
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
            occupancy_counts = np.bincount(behave_var.astype(int), minlength=n_position_bins)
            occupancy_map = occupancy_counts / np.sum(occupancy_counts)

        onsets = np.where(np.diff(time_mask.astype(int)) == 1)[0]
        offsets = np.where(np.diff(time_mask.astype(int)) == -1)[0]

        for onset, offset in zip(onsets, offsets):
            spike_counts = sorted_fcm[onset:offset,:]
            n_time_bins = spike_counts.shape[0]

        self.replay.bayesian_posterior = bayesian_decoder(firing_rate_map, spike_counts, occupancy_map, time_bin_width, n_time_bins, n_position_bins)
        self.replay.radon_score, self.replay.radon_angle = calculate_radon_score(self.replay.bayesian_posterior)
        self.replay.linear_corr = calculate_linear_weighted_correlation(self.replay.bayesian_posterior)
        self.replay.R_max, self.replay.V_max, self.replay.rho_max, self.replay.R_map = calculate_custom_replay_score(self.replay.bayesian_posterior, time_bin_width, position_bin_edges, fract_thresh, V_range, rho_range)

    def state_space_decoder(self, train_mask, train_behave, test_mask, test_behave):
        """Compute state space decoding of activity during identified windows and compare to template sequence.
        Based on Denovellis, ..., Frank, 2021
        This function just ensures all data is processed correctly and saved to file for use in their package.
        TODO: do we want to keep some way of tracking where the transitions between windows from time_mask are in the spike data?"""   

        logger.warning("This will preprocess and save the data for state space decoder.")
        # prepare data
        filename = self.replay.savepath + os.sep + "SSdata_bin" + str(self.aefizz.settings.replay_state_space_decoder_bin_size) + ...
        "_train_" + self.aefizz.settings.replay_decoder_train_time_period + "_test_" + self.aefizz.settings.replay_decoder_test_time_period 
        
        if (not os.path.exists(filename + ".npz")) or self.aefizz.settings.redo_compute:

            # prepare training data
            train_spikes, train_time, frame_for_bin = prepare_state_space_decoder_data(self.aefizz.spike_df, train_mask, self.aefizz.session, self.aefizz.settings.replay_state_space_decoder_bin_size)
            # resample behaviour data to match spikes
            dummy = np.zeros(len(self.aefizz.video_df))
            dummy[train_time] = train_behave #  but actually need to populate with behavioral variable that we're trying to decode (e.g. %escape)
            train_position = dummy[frame_for_bin]

            # prepare test data
            test_spikes, test_time, frame_for_bin = prepare_state_space_decoder_data(self.aefizz.spike_df, test_mask, self.aefizz.session, self.aefizz.settings.replay_state_space_decoder_bin_size)
            # resample behaviour data to match spikes
            dummy = np.zeros(len(self.aefizz.video_df))
            dummy[test_time] = test_behave #  but actually need to populate with behavioral variable that we're trying to decode (e.g. %escape)
            test_position = dummy[frame_for_bin]

            # save data
            np.savez(filename + ".npz", 
                     train_spikes=train_spikes, 
                     train_time=train_time,
                     train_mask= train_mask, 
                     train_position=train_position,
                     test_spikes=test_spikes,
                     test_time=test_time,
                     test_mask= test_mask,
                     test_position=test_position)
            
            np.savez(filename + "_settings.npz", 
                     settings=asdict(self.aefizz.settings))

        logger.warning("State space decoder data saved to " + filename + ".npz" + " . Now run the state space decoder in behave_analysis > analyze > replay > SSdecoder.ipynb.")

