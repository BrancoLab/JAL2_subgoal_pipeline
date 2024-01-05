import os

def make_directory(path):
    """Makes a directory if it doesn't already exist to save files to"""
    if not os.path.isdir(path):
        os.makedirs(path)
    assert os.path.isdir(path), "Directory not created"
    return path