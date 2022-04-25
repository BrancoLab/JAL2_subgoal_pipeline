from behave_analysis.analyze.analysis_funcs import get_escape_initiation_idx, get_which_side

def trial_is_eligible(self, onset_frames: list) -> bool:
    eligible = (self.stim_type in ['homing', 'threshold_crossing'] \
                    or (successful_escape(self, onset_frames) \
                        and escape_starts_near_threat_zone(self, onset_frames[0]) \
                        and self.num_successful_escapes_this_session <  self.settings.max_num_trials)) \

    return eligible

def successful_escape(self, onset_frames: list) -> bool:
    if self.stim_type in ['laser', 'homing', 'threshold_crossing']:           return True
    if self.tracking_data['distance rel. to shelter'][onset_frames[0]] < 500: return False # if trial starts near shelter (within 50cm), it's not a real trial #! other users remove
    if escape_completed_in_time(self, onset_frames) and escape_starts_near_threat_zone(self, onset_frames[0]):
        self.num_successful_escapes_this_session += 1
        return True
    else:
        return False

def escape_completed_in_time(self, onset_frames: list) -> bool:
    completed_in_time = (self.tracking_data['distance rel. to shelter'][onset_frames[0]:onset_frames[0]+self.fps*self.settings.max_escape_duration] \
                        < self.settings.min_distance_from_shelter*self.session.video.pixels_per_cm).any()
    return completed_in_time

def escape_starts_near_threat_zone(self, trial_start_idx: int)->bool:
    RT = get_escape_initiation_idx(self, trial_start_idx)
    y_at_escape_initiation = self.tracking_data['avg_loc'][trial_start_idx+RT, 1]
    escape_initiation_near_threat_zone = y_at_escape_initiation < self.session.video.registration_size[1]/2 - 100
    return escape_initiation_near_threat_zone
