# Custom Lib
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.utils.collect_session_IDs import collect_session_IDs_analysis
from settings.settings_analyze import settings_analyze as settings_a
from databank import databank
from behave_analysis.process.process import Process

session_IDs = collect_session_IDs_analysis(settings_a.analysis, databank)
for session_ID in session_IDs:
    session = Process(session_ID).load_session()
    open_tracking_data(session)
    pos_data = {}
    pos_data['trajectory x'] = self.tracking_data['avg_loc'][:, 0]
    pos_data['trajectory y'] = self.tracking_data['avg_loc'][:, 1]