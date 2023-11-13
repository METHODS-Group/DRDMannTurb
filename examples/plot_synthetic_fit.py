"""
==================
Synthetic Data Fit
==================

"""
import torch
import torch.nn as nn

from drdmannturb import EddyLifetimeType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

# %%

# from drdmannturb.shared.parameters import (
# LossParameters,
# NNParameters,
# PhysicalParameters,
# ProblemParameters,
# )

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

L = 0.59

Gamma = 3.9
sigma = 3.2

domain = torch.logspace(-1, 2, 20)

# %%
pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.ReLU(), nn.ReLU()],
    ),
    prob_params=ProblemParameters(
        nepochs=5, learn_nu=True, eddy_lifetime=EddyLifetimeType.TAUNET
    ),
    loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-2),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    device=device,
)

# %%
k1_data_pts = domain
DataPoints = [(k1, 1) for k1 in k1_data_pts]

# %%
Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

# %%
pb.eval(k1_data_pts)
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot()

# %%
# import matplotlib.pyplot as plt

# plt.figure()

# plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
# plt.legend()
# plt.xlabel("Epoch Number")
# plt.ylabel("MSE")
# plt.yscale("log")

# plt.show()


# %% [markdown]
# ### Save Model with Problem Metadata

# %%
pb.save_model("../results/")

# %% [markdown]
# ### Loading Model and Problem Metadata

# %%
import pickle

# TODO: fix this data load and parameter equivalence check
path_to_parameters = "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"

with open(path_to_parameters, "rb") as file:
    (
        nn_params,
        prob_params,
        loss_params,
        phys_params,
        model_params,
    ) = pickle.load(file)

# %% [markdown]
# ### Recovering Old Model Configuration and Old Parameters

# %%
pb_new = CalibrationProblem(
    nn_params=nn_params,
    prob_params=prob_params,
    loss_params=loss_params,
    phys_params=phys_params,
    device=device,
)

pb_new.parameters = model_params

assert (pb.parameters == pb_new.parameters).all()
