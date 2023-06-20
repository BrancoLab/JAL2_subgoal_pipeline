from pathlib import Path
from behave_analysis.database.Mice.mouse_class import Mouse

# Mices
JAL002 = Mouse(
    nick_name="JAL002",
    total_sessions=0,
    mouse_number_pyrat="xxx-xxxxxxx",
    experiment_file_names=None,
    root_path=Path(r"D:\efizz\JAL002"),
    # root_path=Path(r"C:\Users\jreggiani\Documents\Experimental_Data\JAL002"),
)
