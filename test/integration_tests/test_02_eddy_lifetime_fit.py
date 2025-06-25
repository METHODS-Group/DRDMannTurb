"""
Test the functionality and results of the eddy lifetime fit example with an NN.

This tests the synthetic data fitting functionality using TAUNET eddy lifetime type.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
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


@pytest.fixture(scope="module")
def device():
    """Set up device for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.set_default_dtype(torch.float64)
    return device


@pytest.fixture(scope="module")
def physical_parameters():
    """Set up physical parameters for testing."""
    zref = 90  # reference height
    z0 = 0.02
    uref = 11.4
    ustar = 0.556  # friction velocity

    L = 0.593 * zref  # length scale
    Gamma = 3.89  # time scale
    sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # magnitude (σ = αϵ^{2/3})

    k1 = torch.logspace(-1, 2, 20, dtype=torch.float64) / zref

    return {
        "zref": zref,
        "z0": z0,
        "uref": uref,
        "ustar": ustar,
        "L": L,
        "Gamma": Gamma,
        "sigma": sigma,
        "k1": k1,
    }


@pytest.fixture(scope="module")
def calibration_problem(device, physical_parameters):
    """Set up calibration problem for testing."""
    params = physical_parameters

    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            hidden_layer_sizes=[10, 10],
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(
            nepochs=10,
            learn_nu=False,
            eddy_lifetime=EddyLifetimeType.TAUNET,
            num_components=4,
        ),
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(
            L=params["L"], Gamma=params["Gamma"], sigma=params["sigma"], ustar=params["ustar"], domain=params["k1"]
        ),
        logging_directory="runs/synthetic_fit",
        device=device,
    )
    return pb


@pytest.fixture(scope="module")
def test_data(physical_parameters):
    """Generate test data for calibration."""
    params = physical_parameters
    return OnePointSpectraDataGenerator(data_points=params["k1"], zref=params["zref"], ustar=params["ustar"]).Data


class TestPhysicalParameters:
    """Test physical parameter calculations."""

    def test_length_scale_calculation(self, physical_parameters):
        """Test that length scale is calculated correctly."""
        params = physical_parameters
        expected_L = 53.37
        assert np.isclose(params["L"], expected_L, atol=1e-2), f"Expected L={expected_L}, got L={params['L']}"

    def test_time_scale_calculation(self, physical_parameters):
        """Test that time scale is calculated correctly."""
        params = physical_parameters
        expected_Gamma = 3.89
        assert np.isclose(
            params["Gamma"], expected_Gamma, atol=1e-2
        ), f"Expected Gamma={expected_Gamma}, got Gamma={params['Gamma']}"

    def test_magnitude_calculation(self, physical_parameters):
        """Test that magnitude is calculated correctly."""
        params = physical_parameters
        expected_sigma = 0.0493
        assert np.isclose(
            params["sigma"], expected_sigma, atol=1e-2
        ), f"Expected sigma={expected_sigma}, got sigma={params['sigma']}"


class TestCalibrationProblem:
    """Test calibration problem setup and execution."""

    def test_calibration_problem_initialization(self, calibration_problem):
        """Test that calibration problem initializes correctly."""
        pb = calibration_problem
        assert pb is not None
        assert hasattr(pb, "OPS")
        assert hasattr(pb, "parameters")

    def test_calibration_execution(self, calibration_problem, test_data):
        """Test that calibration runs successfully."""
        pb = calibration_problem

        # Run calibration
        optimal_parameters = pb.calibrate(data=test_data)

        # Check that calibration produced results
        assert optimal_parameters is not None
        assert hasattr(optimal_parameters, "L")
        assert hasattr(optimal_parameters, "Gamma")
        assert hasattr(optimal_parameters, "sigma")

    def test_calibrated_parameters_values(self, calibration_problem, test_data):
        """Test that calibrated parameters are within expected ranges."""
        pb = calibration_problem
        optimal_parameters = pb.calibrate(data=test_data)

        # Test calibrated values (these might need adjustment based on your expected results)
        expected_L = 51.6535
        expected_Gamma = 1.5522
        expected_sigma = 0.0483

        assert np.isclose(
            optimal_parameters["L"], expected_L, atol=1e-2
        ), f"Expected L={expected_L}, got L={optimal_parameters['L']}"
        assert np.isclose(
            optimal_parameters["Γ"], expected_Gamma, atol=1e-2
        ), f"Expected Gamma={expected_Gamma}, got Gamma={optimal_parameters['Γ']}"
        assert np.isclose(
            optimal_parameters["σ"], expected_sigma, atol=1e-3
        ), f"Expected sigma={expected_sigma}, got sigma={optimal_parameters['σ']}"


class TestModelSaveLoad:
    """Test model saving and loading functionality."""

    def test_model_save_and_load(self, calibration_problem, test_data):
        """Test that model can be saved and loaded correctly."""
        pb = calibration_problem

        # Run calibration
        pb.calibrate(data=test_data)

        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save model
            pb.save_model(temp_path)

            # Construct expected save path
            save_path = temp_path / "EddyLifetimeType.TAUNET_DataType.KAIMAL.pkl"

            # Check that file was created
            assert save_path.exists(), f"Model file not created at {save_path}"

            # Load model
            with open(save_path, "rb") as file:
                (
                    nn_params,
                    prob_params,
                    loss_params,
                    phys_params,
                    model_params,
                ) = pickle.load(file)

            # Create new calibration problem from loaded parameters
            pb_new = CalibrationProblem(
                nn_params=nn_params,
                prob_params=prob_params,
                loss_params=loss_params,
                phys_params=phys_params,
                device=pb.device,
            )

            # Set loaded parameters
            pb_new.parameters = model_params

            # Check that parameters match
            assert np.ma.allequal(
                pb.parameters, pb_new.parameters
            ), "Loaded model parameters don't match original parameters"


class TestPlotting:
    """Test plotting functionality (if available)."""

    @pytest.mark.skipif(not hasattr(CalibrationProblem, "plot"), reason="Plotting functionality not available")
    def test_plot_generation(self, calibration_problem, test_data):
        """Test that plots can be generated without errors."""
        pb = calibration_problem
        pb.calibrate(data=test_data)

        # Test plot generation (should not raise exceptions)
        try:
            pb.plot()
            pb.plot_losses(run_number=0)
        except Exception as e:
            pytest.fail(f"Plotting failed with exception: {e}")


# Optional: Mark slow tests
@pytest.mark.slow
class TestLongRunningCalibration:
    """Tests that take longer to run."""

    def test_extended_calibration(self, calibration_problem, test_data):
        """Test calibration with more epochs for better convergence."""
        pb = calibration_problem

        # Modify to run more epochs
        pb.prob_params.nepochs = 50

        optimal_parameters = pb.calibrate(data=test_data)

        # Check for better convergence with more epochs
        assert optimal_parameters is not None


# Optional: Parametrized tests for different configurations
@pytest.mark.parametrize("learn_nu", [True, False])
@pytest.mark.parametrize("num_components", [3, 4, 6])
def test_different_configurations(device, physical_parameters, learn_nu, num_components):
    """Test calibration with different configuration parameters."""
    # TODO: This does not pass????
    params = physical_parameters

    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            hidden_layer_sizes=[10, 10],
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(
            nepochs=5,  # Reduced for faster testing
            learn_nu=learn_nu,
            eddy_lifetime=EddyLifetimeType.TAUNET,
            num_components=num_components,
        ),
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(
            L=params["L"], Gamma=params["Gamma"], sigma=params["sigma"], ustar=params["ustar"], domain=params["k1"]
        ),
        device=device,
    )

    test_data = OnePointSpectraDataGenerator(data_points=params["k1"], zref=params["zref"], ustar=params["ustar"]).Data

    # Should not raise exceptions
    optimal_parameters = pb.calibrate(data=test_data)
    assert optimal_parameters is not None
