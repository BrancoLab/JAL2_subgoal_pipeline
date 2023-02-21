import os
import dill as pickle

def open_tracking_data(self):
        
        file = os.path.join(self.session.file_path, "fully_processed_tracking_data.pickle")
        assert file, 'Tracking data not found for session: {}'.format(self.session.name)
        with open(file, "rb") as dill_file: 
                self.tracking_data = pickle.load(dill_file)
        