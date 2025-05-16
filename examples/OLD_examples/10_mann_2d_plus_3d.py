"""
================================================
Example 10: 2D Low Frequency + 3D High Frequency
================================================

This example demonstrates the 2d+3d modeling available through the :py:class:`FluctuationFieldGenerator` class.
For simplicity, we use the standard Mann model for the 3d field, however this is compatible with any of the models.

The documentation for the :py:class:`LowFreqGenerator` class contains more detail on the 2d modeling, but, briefly,
the model is inspired by the Von Karman model. The generation of the 2d+3d field is not much more complicated than
a 3d field alone, however it may be more memory-intensive depending on the resolution provided.

Generating a box from a 2d+3d model requires simulating the 2d field first. Then, per 3d box, we obtain the 3d field's
physical coordinates (by default, centered on the :math:`y`-axis of the 2d field) and interpolate the 2d field onto
this finer grid. Since we assume that the 2d field is vertically invariant and statistically independent of the 3d
field, we can simply add the 2d field to each vertical slice of the 3d field to obtain the complete result.
"""

import numpy as np

from drdmannturb.fluctuation_generation.fluctuation_field_generator import FluctuationFieldGenerator

#######################################################################################
# 2d parameters
# -------------
#
# Currently, the parameters for the 2d model should be provided as a dictionary.
config_2d = {
    "sigma2": 2.0,
    "L_2d": 15_000.0,
    "z_i": 500.0,
    "psi": np.deg2rad(43.0),
    "L1_factor": 8,
    "L2_factor": 8,
    "exp1": 11,
    "exp2": 11,
}

#######################################################################################
# 3d parameters
# -------------
#
# These are identical to the parameters used in Example 8.
z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
plexp = 0.2
windprofiletype = "PL"

L = 0.593 * zref
Gamma = 3.89
sigma = 0.052

Lx = 720
Ly = 64
Lz = 64

nBlocks = 2
grid_dimensions = np.array([Lx / 4, Ly, Lz])

grid_levels = np.array([6, 4, 4])
Type_Model = "Mann"

#######################################################################################
# Construct the generators
# ------------------------
#
# Now, we're going to construct a 3d and a 2d+3d generator here, for the sake of
# comparison. These will be identical besides the provision of the ``config_2d_model``
# parameter to the 2d+3d generator.

generator_3d = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    seed=1,
    config_2d_model=None,
)

generator_2d3d = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    seed=1,
    config_2d_model=config_2d,
)

#######################################################################################
# Generate the boxes
# ------------------
#
# Now, we generate the boxes as usual.

box_3d = generator_3d.generate(nBlocks, zref, uref, z0, windprofiletype, plexp)
box_2d3d = generator_2d3d.generate(nBlocks, zref, uref, z0, windprofiletype, plexp)
