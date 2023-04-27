from dataclasses import dataclass
from behave_analysis.database.Mice.mouse_class import Mouse

@dataclass(frozen=True)
class Experiment(Mouse):
    """A class to store experiment by experiment information"""
    experiment_name: str
    experiment_idx: int # E.G if this was Mushroom 1, this would be 1
    experiment_date: str
    experiment_time: str
    experiment_path: str # Just the name of the experiment folder e.g. 001_mushroom1_2023_03_10T07_15_15
    shelter_time: list  # When was shelter placed in seconds
    barrier_time: list # When was barrier placed in seconds