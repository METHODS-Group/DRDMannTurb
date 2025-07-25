"""Script for fitting the coherence and 6 component data."""

from pathlib import Path

import torch
import torch.nn as nn

import drdmannturb as drdmt
import drdmannturb.spectra_fitting.spectral_tensor_models as stm

# NOTE: Below is obtained from the LES data... we used the 31st of 60
#       heights that we were given, so this is height[30]
zref = 148.56202535609793

spectra_file = Path("data_cleaned/STORM_downsampled_one_point_spectra.csv")
coherence_file = Path("data_cleaned/STORM_FULL_FIDELITY_coherence_data.csv")

data_loader = drdmt.CustomDataLoader(ops_data_file=spectra_file, coherence_data_file=coherence_file)

pb = drdmt.CalibrationProblem(
    data_loader=data_loader,
    model=stm.RDT_SpectralTensor(
        eddy_lifetime_model=stm.TauNet_ELT(
            taunet=stm.TauNet(
                n_layers=5,
                hidden_layer_sizes=[15, 20, 20, 20, 15],
                activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                learn_nu=False,
            )
        ),
        energy_spectrum_model=stm.Learnable_ESM(
            p_init=5.0,
            q_init=3.0,
        ),
        L_init=6.0,  # Should be 70?
        gamma_init=3.0,  # Should be 3.7?
        sigma_init=0.25,  # Should be 0.04?????
    ),
    loss_params=drdmt.LossParameters(
        alpha_pen1=1.0,
        alpha_pen2=1.0,
        beta_reg=1e-2,
        gamma_coherence=1.25,
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

pb.calibrate(
    optimizer_class=torch.optim.Adam,
    lr=1.0,
    optimizer_kwargs={
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-5,
    },
)

pb.plot()
