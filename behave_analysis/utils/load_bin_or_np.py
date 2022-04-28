# Import OS libs
import numpy as np
from pathlib import Path
from loguru import logger
import os

def load_or_open(path_file_location: str, data_type: str, bin_file: Path, idx: int, **kwargs):
    """This function checks if a np version of the bin file exsists. If not, it calls a func to create
    one. This is due to slow speed associated within loading entire bin files.

    Args:
        path_file_location (str): The path string of the file your wishing to open. Bonsai or imec.
        data_type (str): _description_
        bin_file (Path): _description_
        idx (int): _description_

    Returns:
        _type_: _description_
    """
    savepath = Path(path_file_location).parent / (Path(path_file_location).stem + f"_{data_type}_sync.npy")
    if savepath.exists():
        logger.debug("Loading a previously converted bin file from a npy file")
        return np.load(savepath)
    else:
        logger.debug("Opening a binary and saving to numpy to save time on opening")
        binary = load_bin(str(bin_file), **kwargs)
        signal = binary[:, idx].copy()
        np.save(savepath, signal)
        return signal

def load_bin(filepath, nsigs=385, dtype=None, order=None):
    """Loads and reshapes a bin file to a memory-map.

    Args:
        filepath (str): The path string of the bin file for bonsai or imec
        nsigs (int):  No idea why 385, there are 348 channels so not sure where this comes from
        dtype (str): The data-type used to interpret the file contents
        order (str): Store data in row-major order "C" or column-major order "F"

    Returns:
        _type_: A of <class 'numpy.memmap'> of 2D shape (samples, 385).
    """
    filepath = Path(filepath)
    assert os.path.isfile(filepath), "Path is a folder, not a file"
    logger.debug(f'Opening BIN file: "{filepath}" ({os.path.getsize(filepath)})')
    dtype = dtype or np.float64
    order = order or "C"
    assert os.path.isfile(file), "Path is a folder, not a file"
    with open(filepath, "r") as fin:
        data = np.memmap(fin, dtype=dtype, order=order, mode="r")
    return data.reshape(-1, nsigs) #Reshape to [unknown dimension, nsigs]

file = "C:/Users/JoannaA/Desktop/data/ephys/test0_g0_imec0/test0_g0_t0.imec0.ap.bin"
data = load_bin(file)

