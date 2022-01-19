def print_settings(settings: object):
    for key in settings.__dict__.keys():
        if settings.__dict__[key] and not key in ['by_experiment', 'experiments', 'by_session', 'sessions']:
            print(" {}: {}".format(key, settings.__dict__[key]))
    if settings.by_experiment:      print(" - experiments: {}".format(settings.experiments))
    if settings.by_session:         print(" - sessions: {}".format(settings.sessions))

def print_settings_analysis(settings: object):
    for key in settings.__dict__.keys():
        if settings.__dict__[key] and not key in ['analysis']:
            print(" {}: {}".format(key, settings.__dict__[key]))
    settings = settings.analysis
    if not settings.compare:
        if settings.by_experiment:      print(" - experiments: {}".format(settings.experiments))
        if settings.by_session:         print(" - sessions: {}".format(settings.sessions))
    if settings.compare:
        for group_num in range(1,9):
            key = 'group_' + str(group_num)
            if settings.__dict__[key] is None: continue
            print(" - group {}: {}".format(group_num, settings.__dict__[key]))