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
from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept

from behave_analysis.database.Experiments.JAL004_ex import JAL4_mush1, JAL4_3rdSept, JAL4_19thSept, JAL4_11thSept

from behave_analysis.database.Experiments.JAL005_ex import JAL5_mush1, JAL005_8thSept, JAL005_21stSept

from behave_analysis.database.Experiments.JAL006_ex import (
    JAL6_hab_1mar,
    JAL6_shelt_4mar,
    JAL6_28mar,
    JAL6_flip5_25mar,
    JAL6_flip7_1apr,
    JAL6_flip4_21mar,
    JAL6_flip3_18mar,
)

from behave_analysis.database.Experiments.JAL007_ex import (
    JAL7_hab_1mar,
    JAL7_empty_shelter_5mar,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip2_12mar,
    JAL7_flip5_22mar,
    JAL7_23apr,
    JAL7_30apr,
)

from behave_analysis.database.Experiments.JAL008_ex import (
    JAL8_shelt_22apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_tiny_3may,
    JAL8_flip4_10may,
    JAL8_14may,
)

# ## -------------JR BEHAVIOR MICE

from behave_analysis.database.Experiments.Burrow_ex import burrow_3456, burrow_3440, burrow_3457, burrow_3457_2

# Commonly used experiments
# experiments_objects = [JAL6_28mar]
experiments_objects = [JAL6_flip3_18mar]
# experiments_objects = [JAL4_3rdSept]
# experiments_objects = [JAL7_sesh8_9apr]
# experiments_objects = [JAL8_14may]
# experiments_objects = [JAL6_flip5_25mar]
# experiments_objects = [JAL7_flip2_12mar]
# experiments_objects = [JAL8_flip1_25apr]

# Grouped experiments
# JAL6
# experiments_objects = [JAL6_28mar, JAL6_flip5_25mar, JAL6_flip4_21mar, JAL6_flip3_18mar, JAL6_flip7_1apr]

# JAL7
# experiments_objects = [JAL7_sesh8_9apr, JAL7_sesh9_16apr, JAL7_23apr, JAL7_30apr]

# JAL8
# experiments_objects = [JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip4_10may, JAL8_14may]

# HPC: [flip4stSept_003, JAL3_flip_rot, JAL4_11thSept]
# postprocess: [flip4stSept_003]
# process after HPC: [JAL6_flip7_1apr, JAL6_flip3_18mar] #  these have bugs!
# LDA with linshit = [JAL8_shelt_22apr]
# admire plots: [JAL7_empty_shelter_5mar]
# redoLDA: [JAL3_flip_rot] [flip4stSept_003] # exclude proximal
