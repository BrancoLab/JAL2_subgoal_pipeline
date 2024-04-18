from pathlib import Path
from behave_analysis.database.Mice.mouse_class import Mouse

# Mices

JAL001 = Mouse(
    nick_name="JAL001",
    total_sessions=9,
    mouse_number_pyrat="BAA-1102922",
    experiment_file_names=None,
    root_path = Path(r"JAL001")
)

JAL002 = Mouse(
    nick_name="JAL002",
    total_sessions=9,
    mouse_number_pyrat="xxx-xxxxxxx",
    experiment_file_names=None,
    root_path=Path(r"JAL002")
)

JAL003 = Mouse(
    nick_name="JAL003",
    total_sessions=11,
    mouse_number_pyrat="BAA-1103439",
    experiment_file_names=None,
    root_path = Path(r"JAL003")
)

JAL004 = Mouse(
    nick_name="JAL004",
    total_sessions=10,
    mouse_number_pyrat="BAA-1103424",
    experiment_file_names=None,
    root_path = Path(r"JAL004")
)

JAL005 = Mouse(
    nick_name="JAL005",
    total_sessions=9,
    mouse_number_pyrat="BAA-1103523",
    experiment_file_names=None,
    root_path = Path(r"JAL005")
)

JAL006 = Mouse(
    nick_name="JAL006",
    total_sessions=9,
    mouse_number_pyrat="BAA-1104292",
    experiment_file_names=None,
    root_path = Path(r"JAL006")
)

JAL007 = Mouse(
    nick_name="JAL007",
    total_sessions=9,
    mouse_number_pyrat="BAA-1104293",
    experiment_file_names=None,
    root_path = Path(r"JAL007")
)

## -------------JR BEHAVIOR MICE

JR3440 = Mouse(
    nick_name="JR3440",
    total_sessions=1,
    mouse_number_pyrat="BAA-3440",
    experiment_file_names=None,
    root_path = Path(r"Burrow_test\JR3440"))

JR3456 = Mouse(
    nick_name="JR3456",
    total_sessions=1,
    mouse_number_pyrat="BAA-3456",
    experiment_file_names=None,
    root_path = Path(r"Burrow_test\JR3456"))

JR3457 = Mouse(
    nick_name="JR3457",
    total_sessions=1,
    mouse_number_pyrat="BAA-3457",
    experiment_file_names=None,
    root_path = Path(r"Burrow_test\JR3457"))