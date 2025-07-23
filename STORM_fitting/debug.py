"""Script for fitting the coherence and 6 component data."""

from pathlib import Path

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

spectra_file = Path("data_cleaned/STORM_downsampled_one_point_spectra.csv")
coherence_file = Path("data_cleaned/STORM_FULL_FIDELITY_coherence_data.csv")

data_loader = drdmt.CustomDataLoader(ops_data_file=spectra_file, coherence_data_file=coherence_file)

# Define Calibration Problem
pb = drdmt.CalibrationProblem(
    data_loader=data_loader,
    nn_params=drdmt.NNParameters(
        nlayers=5,
        hidden_layer_sizes=[15, 20, 20, 20, 15],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
    ),
    prob_params=drdmt.ProblemParameters(
        tol=1e-9,
        learn_nu=False,
        learning_rate=0.05,
        use_learnable_spectrum=False,
        p_exponent=5.0,
        q_exponent=3.0,
    ),
    loss_params=drdmt.LossParameters(
        alpha_pen1=0.0,
        alpha_pen2=0.0,
        beta_reg=1e-2,
        gamma_coherence=1.25,
    ),
    phys_params=drdmt.PhysicalParameters(
        L=1.0,
        Gamma=3.0,
        sigma=1600.0,
        Uref=21.0,
        zref=zref,
        ustar=1.0,
        domain=domain,
    ),
    integration_params=drdmt.IntegrationParameters(
        ops_log_min=-4.0,
        ops_log_max=5.0,
        ops_num_points=150,  # TODO:TODO:
        coh_log_min=-4.0,
        coh_log_max=5.0,
        coh_num_points=300,
    ),
    logging_directory="runs/custom_data",
    device="cpu",
)

optimal_params = pb.calibrate(
    optimizer_class=torch.optim.Adam,
    max_epochs=20,
    fix_params=[],
)

pb.plot()
