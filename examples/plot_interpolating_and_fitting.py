"""
======================================
Interpolating Spectra Data and Fitting
======================================

"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
import numpy as np
import torch
import torch.nn as nn

from drdmannturb.enums import DataType
from drdmannturb.interpolation import extract_x_spectra, interpolate
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

path = Path().resolve()
datapath = (
    path / "../docs/source/data" if path.name == "examples" else path / "../data/"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# %% [markdown]
# ## Extract Data from Provided CSVs

# %%
# for interpolation, log10-scaled k1 is used, regular values of the domain used for fitting
L = 0.59
Gamma = 3.9
sigma = 3.2
Uref = 21.0

x_coords_u, u_spectra = extract_x_spectra(datapath / "u_spectra.csv")
x_coords_v, v_spectra = extract_x_spectra(datapath / "v_spectra.csv")
x_coords_w, w_spectra = extract_x_spectra(datapath / "w_spectra.csv")
x_coords_uw, uw_cospectra = extract_x_spectra(datapath / "uw_cospectra.csv")
x_full = [x_coords_u, x_coords_v, x_coords_w, x_coords_uw]
spectra_full = [u_spectra, v_spectra, w_spectra, uw_cospectra]
x_interp, interp_u, interp_v, interp_w, interp_uw = interpolate(
    datapath, num_k1_points=40, plot=True
)
domain = torch.tensor(x_interp)

f = domain
k1_data_pts = 2 * torch.pi * f / Uref

DataPoints = [(k1, 1) for k1 in k1_data_pts]
interpolated_spectra = np.stack((interp_u, interp_v, interp_w, interp_uw), axis=1)

datagen = OnePointSpectraDataGenerator(
    data_points=DataPoints,
    data_type=DataType.AUTO,
    k1_data_points=(
        k1_data_pts.cpu().numpy() if torch.cuda.is_available() else k1_data_pts.numpy()
    ),
    spectra_values=interpolated_spectra,
)

datagen.plot(x_interp, spectra_full, x_full)

# %%

pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.GELU(), nn.GELU()],
    ),
    prob_params=ProblemParameters(nepochs=5),
    loss_params=LossParameters(),
    phys_params=PhysicalParameters(
        L=L, Gamma=Gamma, sigma=sigma, Uref=Uref, domain=domain
    ),
    device=device,
)

# %%


# %%
Data = datagen.Data


# %%
pb.eval(k1_data_pts)
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot(plt_dynamic=False)

# %%
plt.figure()

plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
plt.legend()
plt.xlabel("Epoch Number")
plt.ylabel("MSE")
plt.yscale("log")

plt.show()
