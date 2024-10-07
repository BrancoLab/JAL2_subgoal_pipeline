import os
import socket
import tkinter as tk
from tkinter import filedialog
from loguru import logger

def get_computer_specific_paths(session_path = ''):

    hostname = socket.gethostname()
    base_path = ''

    if hostname == "DESKTOP-AJ2I0CU": # Jasmine machine
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
    
    else: # unknown machine
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        base_path = filedialog.askdirectory(title="Select Directory in which mouse folders are stoared")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        DLC_path = filedialog.askopenfilename(title="Select DLC config.yaml to be used for tracking")

    # check where your folder lives and assign base path accordingly
    # if os.path.exists(os.path.join(my_machine,session_path)):
    #     base_path = my_machine
    if os.path.exists(os.path.join(ceph_path,session_path)):
        base_path = ceph_path
    if os.path.exists(os.path.join(winstor_path,session_path)):
        base_path = winstor_path
    if len(base_path) == 0:
        logger.warning("You sessions is not on winstor or ceph (or you need to reconnect), please check paths!")

    return base_path, DLC_path