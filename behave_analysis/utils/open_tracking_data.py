import os
import dill as pickle

def open_tracking_data(self):
        assert os.path.isfile(self.session.video.tracking_data_file), 'Tracking data not found for session: {}'.format(self.session.name)
        with open(self.session.video.tracking_data_file, "rb") as dill_file: self.tracking_data = pickle.load(dill_file)
        