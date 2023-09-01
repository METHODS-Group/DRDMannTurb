from math import *

import numpy as np
import scipy.special
from scipy.special import kv as Kv
from tqdm import tqdm

# =====================================================================================================
# =====================================================================================================
#
#                                         KERNELS
#
# =====================================================================================================
# =====================================================================================================


#######################################################################################################





# =====================================================================================================
# =====================================================================================================
#
#                                      GAUSSIAN LEVEL-CUT
#
# =====================================================================================================
# 


def MannEddyLifetime(kL):
    return (kL) ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -((kL) ** (-2))))


# =====================================================================================================
