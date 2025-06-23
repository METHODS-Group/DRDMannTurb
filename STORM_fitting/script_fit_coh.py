"""Script for fitting the coherence and 6 component data."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import drdmannturb as drdmt

torch.set_default_dtype(torch.float64)

# Build dataset
domain = torch.logspace(-1, 3, 40)


# NOTE: Below is obtained from the LES data... we used the 31st of 60
#       heights that we were given, so this is height[30]
zref = 148.56202535609793

# NOTE: Empricially obtained from the LES data.
#       As k \to \infty, the Fij data is approximately \propto k^{-5}
k_inf_asymptote = 5.0

spectra_file = Path("data_cleaned/log_downsampled_6component_spectra.dat")
coherence_file = Path("data_cleaned/coherence_data.dat")
CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))

L = 70
Gamma = 3.7
sigma = 0.04
Uref = 21


# # Debug: Check for zeros or very small values
# print("Data shape:", CustomData.shape)
# print("Min values per component:")
# for i in range(1, 7):
#     print(f"  Component {i}: {CustomData[:, i].min().item():.6f}")
# print("Any zeros?", (CustomData[:, 1:] == 0).any().item())
# print("Any negative values?", (CustomData[:, 1:] < 0).any().item())

f = CustomData[:, 0]
k1_data_pts = 2 * torch.pi * f / Uref

gen = drdmt.OnePointSpectraDataGenerator(
    zref=zref,
    data_points=k1_data_pts,
    data_type=drdmt.DataType.CUSTOM,
    spectra_file=spectra_file,
    k1_data_points=k1_data_pts.data.cpu().numpy(),
)

# Debug: Check the generated data
DataPoints, DataValues = gen.Data
# print("\nGenerated data shape:", DataValues.shape)
# print("Generated abs(data) min:", DataValues.abs().min().item())
# print("Generated abs(data) max:", DataValues.abs().max().item())
# print("Any zeros in generated data?", (DataValues == 0).any().item())

# Add small epsilon jic
# DataValues += 1e-10

gen.Data = (DataPoints, DataValues)

domain = torch.logspace(-1, 3, 40)

# NOTE: Below is obtained from the LES data... we used the 31st of 60
#       heights that we were given, so this is height[30]
zref = 148.56202535609793

# Define Calibration Problem
pb = drdmt.CalibrationProblem(
    nn_params=drdmt.NNParameters(
        nlayers=4, hidden_layer_sizes=[10, 10, 10, 10], activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    ),
    prob_params=drdmt.ProblemParameters(
        data_type=drdmt.DataType.CUSTOM,
        tol=1e-9,
        nepochs=2,  # TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:TODO:
        learn_nu=False,
        learning_rate=0.1,
        num_components=6,
    ),
    loss_params=drdmt.LossParameters(
        alpha_pen1=1.5,
        alpha_pen2=1.5,
        beta_reg=1e-2,
    ),
    phys_params=drdmt.PhysicalParameters(
        L=500.0,
        Gamma=13.0,
        sigma=0.25,
        domain=domain,
        Uref=21.0,
        zref=zref,
        use_parametrizable_spectrum=False,
        alpha_low=11.0 / 9.0,  # NOTE: Taken from eye-balling the plot
        alpha_high=-5.0 / 3.0,
        transition_slope=1.0,
        # NOTE: Fij follows k^-5
        # k_inf_asymptote=k_inf_asymptote,
    ),
    logging_directory="runs/custom_data",
    device="cpu",
)

# Add debugging before calibration
print("Testing initial forward pass...")
with torch.no_grad():
    y_test = pb.OPS(k1_data_pts)
    print(f"Initial model output shape: {y_test.shape}")
    print(f"Initial model output range: [{y_test.min().item():.3e}, {y_test.max().item():.3e}]")
    print(f"Any NaN in initial output? {torch.isnan(y_test).any().item()}")
    print(f"Any Inf in initial output? {torch.isinf(y_test).any().item()}")

    # Check the scales
    L_val, Gamma_val, sigma_val = pb.OPS.exp_scales()
    print(f"\nInitial scales: L={L_val:.3f}, Gamma={Gamma_val:.3f}, sigma={sigma_val:.3f}")


# Try fitting
try:
    optimal_params = pb.calibrate(data=gen.Data, coherence_data_file=coherence_file)
except RuntimeError as e:
    print(f"Error during calibration: {e}")

    # Addtl debug
    y = pb.OPS(k1_data_pts)
    print("\nModel output shape: ", y.shape)
    print("Model output min: ", y.min().item())
    print("Model output max: ", y.max().item())

    print("Any NaN in model output?", torch.isnan(y).any().item())
    print("Any Inf in model output?", torch.isinf(y).any().item())

    print("Any NaN in data?", torch.isnan(DataValues).any().item())
    print("Any Inf in data?", torch.isinf(DataValues).any().item())

    print("Any NaN in k1_data_pts?", torch.isnan(k1_data_pts).any().item())


pb.plot()
