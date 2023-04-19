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


# Mices
JAL001 = Mouse(
    nick_name="JAL001",
    total_sessions=3,
    mouse_number_pyrat="BAA-1102922",
    experiment_file_names=None,
    root_path=Path(r"D:\efizz\MouseID_001"),
)
