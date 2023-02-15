'''A meta file that contains all the information about the data paths and the session IDs.'''

###########################################################################################################################

#----Path to the folder where the data is stored-----
databank = {}
databank['path'] = r"D:\efizz" # Where your behavioural data is stored

############################################################################################################################

"""
Arugments:
+ Name of experiment - type: string, e.g. 'ObstacleThenRemove'
+ Previous session - type: bool, e.g. False
+ Folder with data - type: string, e.g. '9692_obstacle_22dec09'
"""

databank['session IDs'] = [[0,0, 'efizz test', False, 'YT6240_23jan20'],
                           [1,0, 'No wall, wall', False, 'YT6240_23jan19']]

#############################################################################################################################

"""
Arugments:
+ Name of experiment - type: string, e.g. 'ObstacleThenRemove'
+ bin file location - type: string, e.g. r"D:\efizz\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap.bin" """

efizz = {"Efizz_test_23Jan20": {"bin": r"D:\efizz\YT6240_23jan20\230120_g0\230120_g0_imec0\230120_g0_t0.imec0.ap.bin",
                                "res": 0},
         "Efizz_test_23Jan19": {"bin": r"D:\efizz\YT6240_23jan19\230119_g0\230119_g0_imec0\230119_g0_t0.imec0.ap.bin",
                                "res": 0}
        }
