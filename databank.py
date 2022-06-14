databank = {}
# databank['path'] = ".\\sample_data\\"                          # sample data
# databank['path'] = "D:\\Dropbox (UCL)\\DAQ\\upstairs_rig\\"      # full dataset
# databank['path'] = r"C:\Users\JoannaA\Dropbox (UCL)\DAQ\upstairs_rig"      # full dataset
# databank['path'] =  r"E:\data\behavior"
databank['path'] = r"D:\behaviour"

databank['session IDs'] = [
                            #----Name of experiment-----prev session------Folder with data------
                            [0,0, 'ObstacleThenRemove',   False, "1677_ObstacleThenRemove_22MAY23"],
                            [1,0, 'NoShelterThenShelter', False, "1677_NoShelterShelter_22MAY31"],
]

#need to refactor with philip if possible, the .res file needs a different entry
# efizz = {"bin": r"D:\Electrophysiology_data\1677_ObstacleThenRemove_22MAY23_g0\1677_ObstacleThenRemove_22MAY23_g0_imec0\1677_ObstacleThenRemove_22MAY23_g0_t0.imec0.ap.bin",
#          "res": r"D:\Electrophysiology_data\1677_ObstacleThenRemove_22MAY23_g0\1677_ObstacleThenRemove_22MAY23_g0_imec0\1677_ObstacleThenRemove_22MAY23_g0_t0.imec0.ap_res.mat"}

efizz = {"bin": r"D:\Electrophysiology_data\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap.bin",
         "res":  r"D:\Electrophysiology_data\1677_NoShelterThenShelter_22MAY31_g0\1677_NoShelterThenShelter_22MAY31_g0_imec0\1677_NoShelterThenShelter_22MAY31_g0_t0.imec0.ap_res.mat"}
 