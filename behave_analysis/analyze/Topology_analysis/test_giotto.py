"""Slower than ripplusplus - Not an option for now"""

from gph import ripser_parallel
import numpy as np
from persim import plot_diagrams

pc = np.random.random((10000, 3))
dgm = ripser_parallel(pc, maxdim=2, n_threads=-1)
print(dgm)
