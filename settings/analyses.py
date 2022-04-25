from behave_analysis.utils.settings_objects import Settings_analyze_local
analyses = {}


# ----------ESCAPES----------------------------------------------

analyses["escapes test"] = \
    Settings_analyze_local( title='escapes trajectories (test)', 
                            plot_escape = True, 
                            by_session=True, by_experiment=False,
                            sessions=[0])

analyses["escapes"] = \
    Settings_analyze_local( title='escapes trajectories', 
                            plot_escape = True, 
                            # experiments=['block edge vectors'])
                            by_session=True, by_experiment=False,
                            all_sessions=True)


# ----------EXPLORE----------------------------------------------

analyses["explore test"] = \
    Settings_analyze_local( title='exploration', 
                            plot_explore = True, 
                            by_session=True, by_experiment=False,
                            sessions=[0])

analyses["explore"] = \
    Settings_analyze_local( title='exploration', 
                            plot_explore = True, 
                            by_session=True, by_experiment=False,
                            all_sessions=True)


# ----------ESCAPE TARGETS----------------------------------------------

analyses["escape targets"] = \
    Settings_analyze_local( title='escapes targets', 
                            plot_targets = True, 
                            experiments=['block edge vectors'])


# ----------SPONTANEOUS HOMINGS------------------------------------

analyses["homings test"] = \
    Settings_analyze_local( title='homings test', 
                            plot_homings = True, 
                            by_session=True, by_experiment=False,
                            sessions=[4])

analyses["homings"] = \
    Settings_analyze_local( title='homings EV', 
                            plot_homings=True,
                            experiments=['block edge vectors'])


# ----------SINGLE TRIALS------------------------------------

analyses["single trial test"] = \
    Settings_analyze_local( title='single trial test', 
                            plot_trial = True, 
                            by_session=True, by_experiment=False,
                            sessions=[1])   

analyses["single trial homing"] = \
    Settings_analyze_local( title='single trial homing', 
                            plot_homing = True, 
                            # by_session=True, by_experiment=False,
                            # sessions=[32])   
                            experiments=['no laser'])   
