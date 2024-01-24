"""Even with GPU acceleration, ripserplusplus does not finish on a small dataset. This limits the use of this package.

It took 2 minutes to run on:
    - 10000 points
    - 4 dimensions
    - 1 homology dimension
    - sparse format

Tried to run on 100k points and it died and ran out of memory after 10 minutes
ending with Failed: cuda error 'out of memory'

NOTE - package not suitable for use in this project unless we focus on smaller time scales such as homings, condition logic will fail as dataset too large

"""

import numpy as np
from ripser import ripser
from persim import plot_diagrams
import numpy as np
from sklearn import decomposition
import ripserplusplus as rpp_py
from loguru import logger
import time

#' record how long it takes to run
start_time = time.time()

logger.info("Running ripser")
d = rpp_py.run("--format point-cloud --dim 1 --sparse", np.random.random((100000, 4)))
print(d)
end_time = time.time()
logger.success("Ripser complete, time took is {}".format(end_time - start_time))
