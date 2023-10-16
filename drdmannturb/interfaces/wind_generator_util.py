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
# 	Fourier Transform of Gaussian Noise
#######################################################################################################


def FourierOfGaussian(noise):
    """Multidimensional Fourier Transform of Gaussian

    Parameters
    ----------
    noise : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    a, b = noise, noise
    for j in range(noise.ndim):
        b = np.roll(np.flip(b, axis=j), 1, axis=j)
    noise_hat = 0.5 * ((a + b) + 1j * (a - b))
    # for j in range(noise.ndim):
    #     np.roll(noise_hat, noise.shape[0] // 2 , axis=j)
    return noise_hat
