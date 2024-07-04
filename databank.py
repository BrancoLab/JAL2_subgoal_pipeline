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
from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept

from behave_analysis.database.Experiments.JAL004_ex import JAL4_mush1, JAL4_3rdSept, JAL4_19thSept, JAL4_11thSept, JAL4_28aug

from behave_analysis.database.Experiments.JAL005_ex import JAL5_3oct, JAL005_8thSept, JAL005_21stSept

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
    JAL8_21may,
)

# ## -------------JR BEHAVIOR MICE

from behave_analysis.database.Experiments.Burrow_ex import burrow_3456, burrow_3440, burrow_3457, burrow_3457_2

experiments_objects = [JAL3_7sept]
# experiments_objects = [JAL3_1sept, JAL3_25aug, JAL3_7sept, JAL3_4sept, 
# JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept,
# JAL005_8thSept, JAL005_21stSept, 
# JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr,
# JAL7_sesh8_9apr, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL7_sesh9_16apr, JAL7_23apr, JAL7_30apr,
# JAL8_flip1_25apr,JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip4_10may, JAL8_14may, JAL8_21may]

# process after HPC: [JAL3_4sept]
# track: [JAL3_25aug, JAL3_7sept, JAL3_4sept]
# postprocess after phy: [JAL3_25aug, JAL3_7sept, JAL3_4sept, JAL6_flip5_25mar]
# process without extractedbin: [JAL3_25aug, JAL3_7sept, JAL3_4sept]

# [JAL3_1sept, JAL4_3rdSept, JAL4_19thSept,JAL005_8thSept,JAL6_28mar,JAL7_sesh8_9apr,JAL8_flip1_25apr,JAL8_flip2_29apr, JAL8_flip4_10may, JAL7_flip5_22mar, JAL7_flip2_12mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL7_sesh9_16apr,JAL4_28aug,JAL7_23apr,JAL7_30apr,JAL8_14may, JAL8_21may]
# LDA with subsampling
# LDA with subsampling and exclude 15cm
# LDA without subsampling
# LDA without subsampling and exclude 15cm
# TODO: double check these work without subsampling
# LDA without subsampling on vectors 
# LDA without subsampling by position
# LDA without subsampling first vs second half TODO: this code doesn't really exist yet
# LDA without subsampling homing based