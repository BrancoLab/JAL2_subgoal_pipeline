import copy
from dataclasses import replace

def settings_overrides(Settings, overrides=None):
    new_settings = copy.deepcopy(Settings)
    if overrides:
        new_settings = replace(new_settings, **{k: v for k, v in overrides.items() if hasattr(new_settings, k)})
    return new_settings