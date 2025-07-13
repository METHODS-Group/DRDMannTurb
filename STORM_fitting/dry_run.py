"""Script for fitting the coherence and 6 component data."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import drdmannturb as drdmt

# Build dataset
domain = torch.logspace(-1, 3, 40)

# NOTE: Below is obtained from the LES data... we used the 31st of 60
#       heights that we were given, so this is height[30]
zref = 148.56202535609793

L = 70
Gamma = 3.7
sigma = 0.04
Uref = 21

spectra_file = Path("data_cleaned/log_downsampled_6component_spectra.dat")
coherence_file = Path("data_cleaned/coherence_data.dat")
CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","), dtype=torch.get_default_dtype())

# Form the one point spectra data.
#
# Note that the file was written out in a CSV with the following format:
#   f, F11(f), F22(f), F33(f), F12(f), F13(f)

# TODO: Double check that the order is correct here.
k1_domain = 2 * torch.pi * CustomData[:, 0] / Uref
ops_data = torch.zeros([len(k1_domain), 3, 3])
ops_data[:, 0, 0] = CustomData[:, 1]
ops_data[:, 1, 1] = CustomData[:, 2]
ops_data[:, 2, 2] = CustomData[:, 3]
ops_data[:, 0, 2] = -1 * CustomData[:, 4]
ops_data[:, 1, 2] = CustomData[:, 5]
ops_data[:, 0, 1] = CustomData[:, 5]

data_dict = {
    "k1": k1_domain,
    "ops": ops_data,
    "coherence": None,
}

# Define Calibration Problem
pb = drdmt.CalibrationProblem(
    nn_params=drdmt.NNParameters(
        nlayers=5,
        hidden_layer_sizes=[15, 20, 20, 20, 15],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
    ),
    prob_params=drdmt.ProblemParameters(
        tol=1e-9,
        nepochs=5,
        learn_nu=True,
        learning_rate=0.3,
        num_components=6,
        use_learnable_spectrum=True,
        p_exponent=5.0,
        q_exponent=3.0,
    ),
    loss_params=drdmt.LossParameters(
        alpha_pen1=1.0,
        alpha_pen2=1.0,
        beta_reg=1e-2,
        gamma_coherence=1.25,
    ),
    phys_params=drdmt.PhysicalParameters(
        L=6.0,
        Gamma=3.0,
        sigma=0.25,
        domain=domain,
        Uref=21.0,
        zref=zref,
        wavenumber_conversion_factor=1 / (torch.pi),
    ),
    integration_params=drdmt.IntegrationParameters(
        ops_log_min=-3.0,
        ops_log_max=3.0,
        ops_num_points=100,
        coh_log_min=-3.0,
        coh_log_max=3.0,
        coh_num_points=300,
    ),
    logging_directory="runs/custom_data",
    device="cpu",
)

optimal_params = pb.calibrate(
    data=data_dict,
    coherence_data_file=coherence_file,
    optimizer_class=torch.optim.Adam,
)

pb.plot()
