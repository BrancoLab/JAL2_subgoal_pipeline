import os
import dill as pickle

def open_tracking_data(session):
        
        file = os.path.join(session.base_path,session.processed_path, "fully_processed_tracking_data.pickle")
        assert file, 'Tracking data not found for session: {}'.format(session.name)
        with open(file, "rb") as dill_file: 
                tracking_data = pickle.load(dill_file)
        
        return tracking_data
        