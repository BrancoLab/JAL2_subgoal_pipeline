import numpy as np

def flatten(iterable: list):
    it = iter(iterable)
    for x in it:
        if isinstance(x, (list, tuple, np.ndarray)):
            for y in flatten(x):
                yield y
        else:
            yield x
