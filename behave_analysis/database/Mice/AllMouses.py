from pathlib import Path
from behave_analysis.database.Mice.mouse_class import Mouse

# Mices

JAL001 = Mouse(
    nick_name="JAL001",
    total_sessions=3,
    mouse_number_pyrat="BAA-1102922",
    experiment_file_names=None,
    root_path = Path(r"JAL001")
)

JAL002 = Mouse(
    nick_name="JAL002",
    total_sessions=0,
    mouse_number_pyrat="xxx-xxxxxxx",
    experiment_file_names=None,
    root_path=Path(r"JAL002")
)

JAL003 = Mouse(
    nick_name="JAL003",
    total_sessions=3,
    mouse_number_pyrat="BAA-1103439",
    experiment_file_names=None,
    root_path = Path(r"JAL003")
)

JAL004 = Mouse(
    nick_name="JAL004",
    total_sessions=3,
    mouse_number_pyrat="BAA-1103424",
    experiment_file_names=None,
    root_path = Path(r"JAL004")
)

JAL005 = Mouse(
    nick_name="JAL005",
    total_sessions=3,
    mouse_number_pyrat="BAA-xxx",
    experiment_file_names=None,
    root_path = Path(r"JAL005")
)