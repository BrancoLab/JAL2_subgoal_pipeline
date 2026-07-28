import os
import socket
import tkinter as tk
from tkinter import filedialog
from loguru import logger

def get_computer_specific_paths(session_path = '', return_ceph = False):

    hostname = socket.gethostname()
    base_path = ''
    ceph_path = ''
    winstor_path = ''
    DLC_path = ''

    if os.getenv("JAL2_DATA_ROOT"):
        ceph_path = os.getenv("JAL2_DATA_ROOT")
        winstor_path = ceph_path
        DLC_path = os.getenv("JAL2_DLC_CONFIG", "")

    elif hostname == "DESKTOP-9CMVP13": # Jasmine machine
        my_machine = r"E:\Experimental_Data"
        ceph_path = r"Z:\Jasmine_Laurence\Experimental_Data"
        winstor_path = r"Y:\Laurence"
        # DLC_path = r"C:\Users\jreggiani\Documents\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
        DLC_path = r"Z:\Jasmine_Laurence\DLC\DLC_220424_JAL6_7_inc\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
    
    elif hostname == "DESKTOP-FBQJ1VU": #Laurence machine
        ceph_path = r"Z:\Jasmine_Laurence\Experimental_Data"
        winstor_path = r"W:\branco\Laurence"
        # my_machine = r"D:\efizz"
        # DLC_path = r"D:\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
        DLC_path = r"Z:\Jasmine_Laurence\DLC\DLC_220424_JAL6_7_inc\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
    
    elif (hostname.startswith("hpc-gw") or hostname.startswith("gpu-sr675-")): # HPC cluster
        ceph_path = r"/ceph/branco/Jasmine_Laurence/Experimental_Data" 
        winstor_path = ceph_path
        DLC_path = r"/ceph/branco/Jasmine_Laurence/DLC/DLC_220424_JAL6_7_inc/JAL_NPX1-Jasmine-2023-03-22/config.yaml"

    else: # unknown machine
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        base_path = filedialog.askdirectory(title="Select Directory in which mouse folders are stoared")
        ceph_path = base_path
        winstor_path = base_path
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        DLC_path = filedialog.askopenfilename(title="Select DLC config.yaml to be used for tracking")

    # check where your folder lives and assign base path accordingly
    # if os.path.exists(os.path.join(my_machine,session_path)):
    #     base_path = my_machine
    if ceph_path and os.path.exists(os.path.join(ceph_path,session_path)):
        base_path = ceph_path
    if winstor_path and os.path.exists(os.path.join(winstor_path,session_path)):
        base_path = winstor_path
    if len(base_path) == 0:
        logger.warning("You sessions is not on winstor or ceph (or you need to reconnect), please check paths!")

    if return_ceph:
        return ceph_path, DLC_path
    else:
        logger.info("All data has been moved from winstor to ceph so we should always load ceph data now")
        return base_path, DLC_path