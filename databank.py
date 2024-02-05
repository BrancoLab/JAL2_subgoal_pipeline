"""
A database of all the experiments and mice run in the JJAL team on the big rig
"""

from behave_analysis.database.Experiments.JAL001_ex import seq3, seq1_2, mush_3
from behave_analysis.database.Experiments.JAL002_ex import (
    firstConnection,
    mushroom1_002,
    seq1_3_002,  # good session
    seq1_2_002,
    mushy4,
    seq1_1,
    mushroom_3_002,
)
from behave_analysis.database.Experiments.JAL003_ex import flip1stSept_003, JAL3_flip_rot, flip4stSept_003

from behave_analysis.database.Experiments.JAL004_ex import (
    JAL4_mush1,
    JAL4_3rdSept,
    JAL4_19thSept,
)

from behave_analysis.database.Experiments.JAL005_ex import JAL5_mush1


# ## -------------JR BEHAVIOR MICE

from behave_analysis.database.Experiments.Burrow_ex import burrow_3456, burrow_3440, burrow_3457, burrow_3457_2

# Currently the code only works with one experiment at a time, so place that experiment in the below list for analysis

experiments_objects = [flip1stSept_003]
