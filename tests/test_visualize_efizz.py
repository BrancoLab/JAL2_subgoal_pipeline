from behave_analysis.database.synthetic_data.generate_data import populate as synthetic_data_generator
from behave_analysis.visualize.visualize_efizz import Visualize_efizz, single_cluster_raster_break_out, extract_trial_spikes_break
import os

# Generate some fake data - Use a fixture
def test_raster():
    synthetic_data = synthetic_data_generator()
    synthetic_data.dataFrame.write_csv(os.getcwd() + "/" + "synthetic_dataframe.csv")
    dic = extract_trial_spikes_break(synthetic_data.dataFrame, onsets = synthetic_data.on_sets)
    single_cluster_raster_break_out(cluster_trial_spikes_dic = dic, onsets = synthetic_data.on_sets)
    x = 4
    assert x == 3, "This test is not implemented yet"

# for session_ID in experiments_objects:
#         session = Process(session_ID).load_session()
#         object = Visualize(session, settings_v).trials(stim_type = 'audio')
        
#         efizz = Visualize_efizz(object, csv_path = os.getcwd() + "/" + "synthetic_dataframe.csv")
#         efizz.load_spike_data()
#         efizz.extract_trial_spikes(stim_type = 'audio', onsets = synthetic_data.on_sets)
#         efizz.single_cluster_raster(stim_type = 'audio')
#         break
        
# # def test_single_cluster_raster():
# #     raise NotImplementedError
    