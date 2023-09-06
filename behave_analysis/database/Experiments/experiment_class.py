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
    shelter_only_time: list  # in minutes when there was only a shelter present e.g. [0, 30] (if until the end of session put -1 as second in list, if barrier comes in after 30 mins. put [0, 30])
    barrier_time: list # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
    barrier_flip_time: int # time in minutes when the barrier was flipped e.g. 184