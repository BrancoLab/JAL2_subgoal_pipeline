from behave_analysis.utils.settings_objects import Settings_analyze_local
analyses = {}

# ----------LASER----------------------------------------------

analyses["laser test"] = \
    Settings_analyze_local( title='laser EV', 
                            plot_laser=True,
                            by_session=True, by_experiment=False,
                            sessions=[1])

analyses["laser"] = \
    Settings_analyze_local( title='laser EV', 
                            plot_laser=True,
                            experiments=['block edge vectors'])

analyses["laser PRE"] = \
    Settings_analyze_local( title='laser PRE', 
                            plot_laser=True,
                            experiments=['block pre edge vectors'])

analyses["laser POST"] = \
    Settings_analyze_local( title='laser POST', 
                            plot_laser=True,
                            experiments=['block post edge vectors'])

analyses["laser PUNISH"] = \
    Settings_analyze_local( title='laser PUNISH', 
                            plot_laser=True,
                            experiments=['block after 2nd edge vector'])

# ----------ESCAPES----------------------------------------------

analyses["escapes test"] = \
    Settings_analyze_local( title='escapes trajectories (test)', 
                            plot_escape = True, 
                            by_session=True, by_experiment=False,
                            sessions=[0])

analyses["escapes"] = \
    Settings_analyze_local( title='escapes trajectories', 
                            plot_escape = True, 
                            experiments=['block edge vectors'])

analyses["escapes PRE"] = \
    Settings_analyze_local( title='escapes trajectories - PRE', 
                            plot_escape = True, 
                            experiments=['block pre edge vectors'])
                    
analyses["escapes POST"] = \
    Settings_analyze_local( title='escapes trajectories - POST', 
                            plot_escape = True, 
                            experiments=['block post edge vectors'])

analyses["escapes PUNISH"] = \
    Settings_analyze_local( title='escapes trajectories - PUNISH', 
                            plot_escape = True, 
                            experiments=['block after 2nd edge vector'])                        

analyses["escapes OPEN"] = \
    Settings_analyze_local( title='escapes trajectories - OPEN', 
                            plot_escape = True, 
                            experiments=['open field'])

# ----------ESCAPE TARGETS----------------------------------------------

analyses["escape targets"] = \
    Settings_analyze_local( title='escapes targets', 
                            plot_targets = True, 
                            experiments=['block edge vectors'])

analyses["escape targets EV vs PRE vs POST"] = \
    Settings_analyze_local( title='escape targets EV vs PRE vs POST', 
                            plot_targets=True,
                            compare=True, 
                            group_1 = ['block edge vectors'], 
                            group_2 = ['block pre edge vectors'], 
                            group_3 = ['block post edge vectors'])

analyses["escape targets all"] = \
    Settings_analyze_local( title='escape targets OPEN vs PRE vs POST vs PUNISH vs NL vs EV', 
                            plot_targets=True,
                            compare=True, 
                            group_1 = ['open field'], 
                            group_2 = ['block pre edge vectors'], 
                            group_3 = ['block post edge vectors'],
                            group_4 = ['block after 2nd edge vector'],
                            group_5 = ['no laser'],    
                            group_6 = ['block edge vectors'])

analyses["escape targets naive vs exp"] = \
    Settings_analyze_local( title='escapes targets naive vs exp', 
                            plot_targets = True, 
                            by_session=True, by_experiment=False,
                            compare=True,
                            group_1 = [0,1,2,3,8,9,10,11,16,17,18,19],
                            group_2 = [4,5,6,7,12,13,14,15,20,21,22,23]
                            )  

analyses["escape targets naive vs exp (EV expts)"] = \
    Settings_analyze_local( title='escapes targets naive vs exp (EV expts)', 
                            plot_targets = True, 
                            by_session=True, by_experiment=False,
                            compare=True,
                            group_1 = [8,9,10,11,16,17,18,19],
                            group_2 = [12,13,14,15,20,21,22,23]
                            )                               

analyses["escape targets naive vs exp all"] = \
    Settings_analyze_local( title='escapes targets naive vs exp all', 
                            plot_targets = True, 
                            by_session=True, by_experiment=False,
                            compare=True,
                            group_1 = [0,1,2,3],
                            group_2 = [4,5,6,7],
                            group_3 = [8,9,10,11],
                            group_4 = [12,13,14,15],
                            group_5 = [16,17,18,19],
                            group_6 = [20,21,22,23],
                            group_7 = [0,1,2,3,8,9,10,11,16,17,18,19],
                            group_8 = [4,5,6,7,12,13,14,15,20,21,22,23])

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

analyses["homings NL"] = \
    Settings_analyze_local( title='homings EV', 
                            plot_homings=True,
                            experiments=['no laser'])                            

analyses["homings PRE"] = \
    Settings_analyze_local( title='homings PRE', 
                            plot_homings=True,
                            experiments=['block pre edge vectors'])

analyses["homings POST"] = \
    Settings_analyze_local( title='homings POST', 
                            plot_homings=True,
                            experiments=['block post edge vectors'])

analyses["homings PUNISH"] = \
    Settings_analyze_local( title='homings PUNISH', 
                            plot_homings=True,
                            experiments=['block after 2nd edge vector'])


# ----------THRESHOLD CROSSINGS------------------------------------

analyses["t xings test"] = \
    Settings_analyze_local( title='t xings test', 
                            plot_t_xings = True, 
                            by_session=True, by_experiment=False,
                            sessions=[1])

analyses["t xings"] = \
    Settings_analyze_local( title='t xings EV', 
                            plot_t_xings=True,
                            experiments=['block edge vectors'])

analyses["t xings NL"] = \
    Settings_analyze_local( title='t xings NL', 
                            plot_t_xings=True,
                            experiments=['no laser'])                            

analyses["t xings PRE"] = \
    Settings_analyze_local( title='t xings PRE', 
                            plot_t_xings=True,
                            experiments=['block pre edge vectors'])

analyses["t xings POST"] = \
    Settings_analyze_local( title='t xings POST', 
                            plot_t_xings=True,
                            experiments=['block post edge vectors'])

analyses["t xings PUNISH"] = \
    Settings_analyze_local( title='t xings PUNISH', 
                            plot_t_xings=True,
                            experiments=['block after 2nd edge vector'])

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
