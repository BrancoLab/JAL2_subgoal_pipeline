import socket
import tkinter as tk
from tkinter import filedialog

def get_computer_specific_paths():

    hostname = socket.gethostname()

    if hostname == "W10-221I": # Jasmine machine
        base_path = r"E:\Experimental_Data"
        DLC_path = r"C:\Users\jreggiani\Documents\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
    elif hostname == "DESKTOP-FBQJ1VU": #Laurence machine
        base_path = r"E:\efizz"
        # base_path = r"D:\efizz"
        DLC_path = r"D:\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml"
    else: # unknown machine
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        base_path = filedialog.askdirectory(title="Select Directory in which mouse folders are stoared")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        DLC_path = filedialog.askopenfilename(title="Select DLC config.yaml to be used for tracking")

    return base_path, DLC_path