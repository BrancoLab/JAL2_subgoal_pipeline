'''
A database of all the experiments and mice run in the JJAL team on the big rig
'''

from behave_analysis.database.Experiments.JAL001_ex import (
    seq3, 
    seq1_2, 
    mush_3
)
from behave_analysis.database.Experiments.JAL002_ex import (
    firstConnection, 
    mushroom1_002, 
    seq1_3_002, 
    seq1_2_002, 
    mushy4, 
    seq1_1, 
    mushroom_3_002
)
from behave_analysis.database.Experiments.JAL003_ex import (
    flip1stSept_003,
)
from behave_analysis.database.Experiments.testymctestface_ex import (
    testbonsaipulse, 
    testbonsaipulse2withefizz, 
    test_NEWgate
)

# Currently the code only works with one experiment at a time, so place that experiment in the below list for analysis
experiments_objects = [flip1stSept_003]