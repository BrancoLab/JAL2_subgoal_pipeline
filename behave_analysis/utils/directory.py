import os
from pathlib import Path 

class Directory():
    def __init__(self, base_folder: str, experiment:str=None, analysis_type: str=None, stim_type: bool=False, tracking_video:bool=False, media_type: str=None):
        if stim_type=='audio':              self.leaf_folder='escape'
        if stim_type=='laser':              self.leaf_folder='laser'
        if stim_type=='homing':             self.leaf_folder='homing'
        if stim_type=='threshold_crossing': self.leaf_folder='threshold crossing'

        if stim_type or 'trial' in analysis_type: self.subfolder = 'trials'
        if analysis_type:
            if 'trajectories' in analysis_type: self.subfolder = 'trajectories'
            if 'targets' in analysis_type:      self.subfolder = 'statistics'
            if 'trajectories' in analysis_type: analysis_type = analysis_type.replace(' trajectories', '')
            if 't xing' in analysis_type:       analysis_type = analysis_type.replace('t xing', 'threshold crossing')
        
        self.base_folder    = base_folder
        self.experiment     = experiment
        self.analysis_type  = analysis_type
        self.stim_video     = stim_type
        self.tracking_video = tracking_video
        self.media_type     = media_type
        self.generate_directory_name()
        self.create_directory()

    def generate_directory_name(self):
        if self.media_type=='plot' and not 'trial' in self.analysis_type:
            self.path = os.path.join(self.base_folder, self.subfolder, self.analysis_type)
        if self.stim_video or 'trial' in self.analysis_type:
            self.path = os.path.join(self.base_folder, self.subfolder, self.experiment, self.leaf_folder)
        if self.tracking_video: 
            self.path = os.path.join(self.base_folder, self.subfolder, "__tracking__", self.leaf_folder)

    def create_directory(self):
        if os.path.isdir(self.path): return
        Path(self.path).mkdir(parents=True)

    def file_name(self, mouse: str=None, trial_num: int=None, minutes_into_session: int=None, title: str=None, color_by: str=None, plot_extension: str = '.png') -> str:
        if self.media_type=='video':
            self.file_name = os.path.join(self.path,"{}-{} ({}').mp4".format(mouse, trial_num+1, minutes_into_session))
        if self.media_type=='plot' and 'trial' in self.analysis_type:
            self.file_name = os.path.join(self.path,"{}-{} ({}'){}".format(mouse, trial_num+1, minutes_into_session, plot_extension))
        if self.media_type=='plot' and not'trial' in self.analysis_type:
            if color_by: color_by_text = '_color=' + color_by
            else       : color_by_text = ''
            self.file_name = os.path.join(self.path, title + color_by_text + plot_extension)

        return self.file_name