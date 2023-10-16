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

from drdmannturb.calibration import CalibrationProblem
from drdmannturb.data_generator import OnePointSpectraDataGenerator
from drdmannturb.interpolation import extract_x_spectra, interp_spectra
from drdmannturb.shared.enums import DataType
from drdmannturb.shared.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)

mpl.style.use("seaborn")

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
x_coords_u, u_spectra = extract_x_spectra(datapath / "u_spectra.csv")
x_coords_v, v_spectra = extract_x_spectra(datapath / "v_spectra.csv")
x_coords_w, w_spectra = extract_x_spectra(datapath / "w_spectra.csv")
x_coords_uw, uw_cospectra = extract_x_spectra(datapath / "uw_cospectra.csv")


# %%
x_interp = np.linspace(
    min(x_coords_w), max(x_coords_w), 40
)  # all coords are on the same here, but choose bounds on the domain which are inclusive of all sampling locations

interp_u = interp_spectra(x_interp, x_coords_u, u_spectra)
interp_v = interp_spectra(x_interp, x_coords_v, v_spectra)
interp_w = interp_spectra(x_interp, x_coords_w, w_spectra)
interp_uw = interp_spectra(x_interp, x_coords_uw, uw_cospectra)

# %%
cmap = plt.get_cmap("Spectral", 4)
custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

plt.plot(
    x_coords_u, u_spectra, "o", label="Observed u Spectra", color=custom_palette[0]
)
plt.plot(x_interp, interp_u, color=custom_palette[0])
plt.plot(
    x_coords_v, v_spectra, "o", label="Observed v Spectra", color=custom_palette[1]
)
plt.plot(x_interp, interp_v, color=custom_palette[1])
plt.plot(
    x_coords_w, w_spectra, "o", label="Observed w Spectra", color=custom_palette[2]
)
plt.plot(x_interp, interp_w, color=custom_palette[2])
plt.plot(
    x_coords_uw,
    uw_cospectra,
    "o",
    label="Observed uw Cospectra",
    color=custom_palette[3],
)
plt.plot(x_interp, interp_uw, color=custom_palette[3])

plt.xlabel(r"$k_1$")
plt.ylabel(r"$k_1 F_i /u_*^2$")
plt.title("Logspace Spectra Interpolation")
plt.legend()

# %%
L = 0.59
Gamma = 3.9
sigma = 3.2
Uref = 21.0

domain_np = np.power(10, x_interp)
domain = torch.tensor(domain_np)

pb = CalibrationProblem(
    nn_params=NNParameters(
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
parameters = pb.parameters
parameters[:3] = [np.log(L), np.log(Gamma), np.log(sigma)]

pb.parameters = parameters[: len(pb.parameters)]

f = domain
k1_data_pts = 2 * torch.pi * f / Uref

DataPoints = [(k1, 1) for k1 in k1_data_pts]
spectra_values = np.stack((interp_u, interp_v, interp_w, -interp_uw), axis=1)

# %%
Data = OnePointSpectraDataGenerator(
    data_points=DataPoints,
    data_type=DataType.AUTO,
    k1_data_points=(
        k1_data_pts.cpu().numpy() if torch.cuda.is_available() else k1_data_pts.numpy()
    ),
    spectra_values=spectra_values,
).Data

# %%
filtered_data_fit = (
    Data[1].cpu().numpy() if torch.cuda.is_available() else Data[1].numpy()
)

plt.plot(
    x_interp,
    filtered_data_fit[:, 0, 0],
    label="Filtered u spectra",
    color=custom_palette[0],
)
plt.plot(
    x_coords_u, u_spectra, "o", label="Observed u Spectra", color=custom_palette[0]
)

plt.plot(
    x_interp,
    filtered_data_fit[:, 1, 1],
    label="Filtered v spectra",
    color=custom_palette[1],
)
plt.plot(
    x_coords_v, v_spectra, "o", label="Observed v Spectra", color=custom_palette[1]
)

plt.plot(
    x_interp,
    filtered_data_fit[:, 2, 2],
    label="Filtered w spectra",
    color=custom_palette[2],
)
plt.plot(
    x_coords_w, w_spectra, "o", label="Observed w Spectra", color=custom_palette[2]
)

plt.plot(
    x_interp,
    filtered_data_fit[:, 0, 2],
    label="Filtered uw cospectra",
    color=custom_palette[3],
)
plt.plot(
    x_coords_uw,
    uw_cospectra,
    "o",
    label="Observed uw Cospectra",
    color=custom_palette[3],
)

plt.title("Filtered and Interpolated Spectra")
plt.xlabel(r"$k_1$")
plt.ylabel(r"$k_1 F_i /u_*^2$")
plt.legend()

# %%
pb.eval(k1_data_pts)
pb.calibrate(data=Data)

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
