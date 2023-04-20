from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Mouse:
    """A class to store mouse by mouse information"""

    nick_name: str
    total_sessions: int
    mouse_number_pyrat: str
    experiment_file_names: list  # A list of the file names for each experiment
    root_path: str  # The path to the local data folder
