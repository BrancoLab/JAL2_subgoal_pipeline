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
from behave_analysis.database.Experiments.JAL003_ex import JAL3_shelt_17aug, JAL3_mush_21aug, JAL3_flip1_22aug, JAL3_flip2_25aug, JAL3_flip3_29aug, JAL3_flip4_1sept, JAL3_flip5_4sept, JAL3_flip6_7sept

from behave_analysis.database.Experiments.JAL004_ex import JAL4_shelt_17aug, JAL4_mush_18aug, JAL4_flip1_21Aug, JAL4_mush2_22Aug, JAL4_flip3_28aug, JAL4_flip4_3Sept, JAL4_flip5_11Sept, JAL4_flip6_19Sept

from behave_analysis.database.Experiments.JAL005_ex import JAL5_shelt_2Sept, JAL5_barr_5Sept, JAL5_flip1_8Sept, JAL5_flip3_21Sept, JAL5_mush_3oct

from behave_analysis.database.Experiments.JAL006_ex import JAL6_hab_1mar, JAL6_shelt_4mar, JAL6_flip3_18mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip6_28mar, JAL6_flip7_1apr, JAL6_flip8_5apr


from behave_analysis.database.Experiments.JAL007_ex import JAL7_hab_1mar, JAL7_empty_shelter_5mar, JAL7_flip2_12mar, JAL7_flip3_15mar, JAL7_flip4_19mar, JAL7_flip5_22mar, JAL7_flip7_4apr, JAL7_flip8_9apr, JAL7_flip9_16apr, JAL7_flip10_23apr, JAL7_tiny_30apr

from behave_analysis.database.Experiments.JAL008_ex import (
    JAL8_shelt_22apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_tiny_3may,
    JAL8_flip3_7may,
    JAL8_flip4_10may,
    JAL8_flip5_14may,
    JAL8_tiny2_21may,
)

# ## -------------JR BEHAVIOR MICE

# from behave_analysis.database.Experiments.Burrow_ex import burrow_3456, burrow_3440, burrow_3457, burrow_3457_2

# literally all sessions!
experiments_objects = [JAL7_hab_1mar,JAL6_hab_1mar]

full_experiments_objects = [JAL3_shelt_17aug, JAL3_mush_21aug, JAL3_flip1_22aug, JAL3_flip2_25aug, JAL3_flip3_29aug, JAL3_flip4_1sept, JAL3_flip5_4sept, JAL3_flip6_7sept,
                       JAL4_shelt_17aug, JAL4_mush_18aug, JAL4_flip1_21Aug, JAL4_mush2_22Aug, JAL4_flip3_28aug, JAL4_flip4_3Sept, JAL4_flip5_11Sept, JAL4_flip6_19Sept,
                       JAL5_shelt_2Sept, JAL5_barr_5Sept, JAL5_flip1_8Sept, JAL5_flip3_21Sept, JAL5_mush_3oct,
                       JAL6_hab_1mar, JAL6_shelt_4mar, JAL6_flip3_18mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip6_28mar, JAL6_flip7_1apr, JAL6_flip8_5apr,
                       JAL7_hab_1mar, JAL7_empty_shelter_5mar, JAL7_flip2_12mar, JAL7_flip3_15mar, JAL7_flip4_19mar, JAL7_flip5_22mar, JAL7_flip7_4apr, JAL7_flip8_9apr, JAL7_flip9_16apr, JAL7_flip10_23apr, JAL7_tiny_30apr,
                       JAL8_shelt_22apr, JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_tiny_3may, JAL8_flip3_7may, JAL8_flip4_10may, JAL8_flip5_14may, JAL8_tiny2_21may]

# experiments_objects = [JAL005_5thSept,
# JAL6_flip4_21mar, JAL6_flip3_18mar,
# JAL8_flip3_7may] # these ones need hdir pkls made! also, maybe it shouldn't be a pkl, but it could be an npz?

"""The complete lists"""

# all barrier flip sessions
flip_experiments_objects = [JAL3_flip1_22aug, JAL3_flip2_25aug, JAL3_flip3_29aug, JAL3_flip4_1sept, JAL3_flip5_4sept, JAL3_flip6_7sept,
                       JAL4_flip1_21Aug, JAL4_flip3_28aug, JAL4_flip4_3Sept, JAL4_flip5_11Sept, JAL4_flip6_19Sept,
                       JAL5_barr_5Sept, JAL5_flip1_8Sept, JAL5_flip3_21Sept,
                       JAL6_flip3_18mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip6_28mar, JAL6_flip7_1apr, JAL6_flip8_5apr,
                       JAL7_flip2_12mar, JAL7_flip3_15mar, JAL7_flip4_19mar, JAL7_flip5_22mar, JAL7_flip7_4apr, JAL7_flip8_9apr, JAL7_flip9_16apr, JAL7_flip10_23apr,
                       JAL8_flip1_25apr, JAL8_flip2_29apr, JAL8_flip3_7may, JAL8_flip4_10may, JAL8_flip5_14may,]

# tiny barrier
tiny_experiments_objects = [JAL8_tiny_3may, JAL8_tiny2_21may, JAL7_tiny_30apr]

# shelter no barrier sessions ["shelter_present",'pre_shelter']
shelter_experiments_objects = [JAL5_shelt_2Sept,JAL4_shelt_17aug, JAL3_shelt_17aug, JAL8_shelt_22apr, JAL7_empty_shelter_5mar, JAL6_shelt_4mar, JAL5_barr_5Sept]
# note: JAL005_5thSept did have an unflipped barrier in there! 

# mushroom
mushroom_experiments_objects = [JAL3_mush_21aug, JAL4_mush_18aug, JAL4_mush2_22Aug, JAL5_mush_3oct]


