import os

def make_directory(path):
    """Makes a directory if it doesn't already exist to save files to"""
    if not os.path.isdir(path):
        os.mkdir(path)
    assert os.path.isdir(path), "Directory not created"