'''A meta file that contains all the information about the data paths and the session IDs.'''

#----Path to the folder where the data is stored-----
databank = {}
databank['path'] = r"D:\Behaviour"

"""
Arugments:
+ Name of experiment - type: string, e.g. 'ObstacleThenRemove'
+ Previous session - type: bool, e.g. False
+ Folder with data - type: string, e.g. '9692_obstacle_22dec09'

test
"""

databank['session IDs'] = [[0,0, 'obstacle', False, '9692_obstacle_22dec09']]


# efizz = {"1677_NoShelterShelter_22MAY31": 
#             {
#             "bin": r"D:\efizz\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap.bin"
#             }
#         }

# efizz = { "1677_NoShelterShelter_22MAY31" : {
#                                               "bin": r"D:\Electrophysiology_data\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap.bin",
#                                                "res":  r"D:\Electrophysiology_data\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap_res.mat"
#                                             }
#         }


# databank['session IDs'] = [
#                             #----Name of experiment-----prev session------Folder with data------
#                             [0,0, 'ObstacleThenRemove',   False, "1677_ObstacleThenRemove_22MAY23"],
#                             [1,0, 'NoShelterThenShelter', False, "1677_NoShelterShelter_22MAY31"],
# ]
