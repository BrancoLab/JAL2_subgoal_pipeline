from pathlib import Path
from behave_analysis.database.Mice.mouse_class import Mouse

# Mices
JAL001 = Mouse(
    nick_name="JAL001",
    total_sessions=3,
    mouse_number_pyrat="BAA-1102922",
    experiment_file_names=None,
    # root_path=Path(r"D:\efizz\MouseID_001"),
    root_path = Path(r"E:\Experimental_Data\JAL001")
)