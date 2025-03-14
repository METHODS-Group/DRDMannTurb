"""
======================================
Example 8: Fluctuation Field Generation
======================================

This example demonstrates the utilities for generating synthetic turbulence, which can be either
from a pre-trained DRD model, or based on some well-known spectra models. ``DRDMannTurb``
provides several utilities for plotting the resulting fields through Plotly, which can be done
in several contexts as well as utilities for saving to VTK for downstream analysis in, e.g.,
ParaView.

"""

#######################################################################################
#   .. centered::
#       This example may take a few seconds to load. Please be patient if using
#       Plotly, as it requires some time to render 3D graphics.
#

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.
from pathlib import Path

import numpy as np
import torch

from drdmannturb.fluctuation_generation import (
    FluctuationFieldGenerator,
)

path = Path().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

#######################################################################################
# Setting Physical Parameters
# ---------------------------
# Here, we set the physical parameters of the environment in which the synthetic wind field is generated:
# the friction velocity :math:`u_\mathrm{red} = 11.4` roughness height :math:`z_0=0.02` and reference height
# of :math:`90`. The physical domain is determined by dimensions in 3D as well as the discretization
# size (grid levels) in each dimension.
z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
plexp = 0.2  # power law exponent
windprofiletype = "PL"  # choosing power law, use log with "LOG" here instead

L = 0.593 * zref
Gamma = 3.89
sigma = 0.052

Lx = 1024
Ly = 256
Lz = 256

nBlocks = 2
grid_dimensions = np.array([Lx, Ly, Lz])

grid_levels = np.array([6, 4, 4])
# grid_levels = np.array([5, 3, 3])
grid_levels = np.array([7, 5, 5])

seed = None

#######################################################################################
# Generating Fluctuation Field from Mann Model
# --------------------------------------------
# Fluctuation fields are generated block-by-block, rather than over the domain entirely.
# Please see section V, B of the original DRD paper for further discussion. Here, we will use 4 blocks.

Type_Model = "Mann"  ### 'Mann', 'VK', 'DRD'

#######################################################################################
# Physical Parameters
# -------------------
# The Mann model requires three parameters, length scale, time scale, and spectrum amplitude scale,
# which are defined above
#
gen_mann = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    model="Mann",
    seed=seed,
    blend_num=0,
)

fluctuation_field = gen_mann.generate(1, zref, uref, z0, windprofiletype, plexp)

print("\n")
print("x var: ", np.var(fluctuation_field[:, :, :, 0]))
print("y var: ", np.var(fluctuation_field[:, :, :, 1]))
print("z var: ", np.var(fluctuation_field[:, :, :, 2]))
