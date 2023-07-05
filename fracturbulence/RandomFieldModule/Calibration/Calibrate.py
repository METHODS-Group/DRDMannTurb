from math import *
import numpy as np
import scipy.optimize
from pylab import *
from time import time


###################################################################################################
#  Calibrate
###################################################################################################


def Calibrate(ObjectiveFunction, **kwargs):
    verbose = kwargs.get("verbose", False)
    tol = kwargs.get("tol", 1.0e-3)

    J = lambda p: ObjectiveFunction(p)
    Jacobian = None  # ObjectiveFunction.Jacobian

    ObjectiveFunction.initialize(**kwargs)
    theta_ini = ObjectiveFunction.get_parameters()
    res = scipy.optimize.minimize(
        J, theta_ini, jac=Jacobian, tol=tol, options={"disp": verbose}
    )
    theta_opt = res.x
    if verbose:
        print(res)
    ObjectiveFunction.finalize(**kwargs)

    return theta_opt


###################################################################################################
###################################################################################################

if __name__ == "__main__":
    """TEST by method of manufactured solutions"""

    pass
