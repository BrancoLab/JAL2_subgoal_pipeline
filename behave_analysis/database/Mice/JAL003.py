from pathlib import Path
from behave_analysis.database.Mice.mouse_class import Mouse

# Mices
JAL003 = Mouse(
    nick_name="JAL003",
    total_sessions=3,
    mouse_number_pyrat="BAA-1103439",
    experiment_file_names=None,
    root_path = Path(r"JAL003")
)