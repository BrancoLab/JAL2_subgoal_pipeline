"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Settings_analyze_efizz:
    run_tunED: bool=True
