"""Tests for SpectralTensorModel save_model/load_model and CalibrationProblem.save_model."""

import numpy as np
import pytest
import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb.nn_modules import TauNet
from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.spectra_fitting import spectral_tensor_models as stm
from drdmannturb.spectra_fitting.data_generator import generate_kaimal_spectra

pytestmark = pytest.mark.unit


def _assert_models_equivalent(model_a: stm.SpectralTensorModel, model_b: stm.SpectralTensorModel):
    """Assert equal state dicts and identical forward outputs on a random wavevector grid."""
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"state_dict mismatch at {key}"

    torch.manual_seed(1)
    k = torch.rand(4, 4, 4, 3) + 0.1
    for phi_a, phi_b in zip(model_a(k), model_b(k)):
        assert torch.equal(phi_a, phi_b)


class TestModelSaveLoad:
    """Round-trip tests for the checkpoint format."""

    def test_taunet_round_trip(self, tmp_path):
        """TauNet architecture (layers, activations, learned nu) survives a round trip."""
        torch.manual_seed(0)
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.TauNet_ELT(
                taunet=TauNet(n_layers=3, hidden_layer_sizes=[8, 12, 8], activations=nn.Tanh(), learn_nu=True)
            ),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=0.59,
            gamma_init=3.9,
            sigma_init=3.2,
        )

        path = model.save_model(tmp_path / "taunet_model")
        assert path.suffix == ".pt" and path.exists()

        loaded = stm.SpectralTensorModel.load_model(path)

        assert isinstance(loaded, stm.RDT_SpectralTensor)
        assert isinstance(loaded.eddy_lifetime_model, stm.TauNet_ELT)
        taunet = loaded.eddy_lifetime_model.tauNet
        assert taunet.n_layers == 3
        assert taunet.hidden_layer_sizes == [8, 12, 8]
        assert all(isinstance(act, nn.Tanh) for act in taunet.activations)
        assert taunet.Ra.fg_learn_nu
        _assert_models_equivalent(model, loaded)

    def test_fixed_nu_round_trip(self, tmp_path):
        """A non-learnable nu is not in the state_dict and must survive via the config."""
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.TauNet_ELT(taunet=TauNet(learn_nu=False, nu_init=-0.5)),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=1.0,
            gamma_init=1.0,
            sigma_init=1.0,
        )
        loaded = stm.SpectralTensorModel.load_model(model.save_model(tmp_path / "m"))

        assert not loaded.eddy_lifetime_model.tauNet.Ra.fg_learn_nu
        assert float(loaded.eddy_lifetime_model.tauNet.Ra.nu) == pytest.approx(-0.5)

    @pytest.mark.parametrize("elt_cls", [stm.Mann_ELT, stm.TwoThirds_ELT, stm.Constant_ELT])
    def test_parameter_free_elts_round_trip(self, tmp_path, elt_cls):
        """Closed-form eddy lifetime models round trip with learned scales intact."""
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=elt_cls(),
            energy_spectrum_model=stm.Learnable_ESM(p_init=3.0, q_init=2.0),
            L_init=0.75,
            gamma_init=2.5,
            sigma_init=1.7,
        )
        loaded = stm.SpectralTensorModel.load_model(model.save_model(tmp_path / "m"))

        assert isinstance(loaded.eddy_lifetime_model, elt_cls)
        assert isinstance(loaded.energy_spectrum_model, stm.Learnable_ESM)
        for name in ("log_L", "log_gamma", "log_sigma"):
            assert torch.equal(getattr(loaded, name), getattr(model, name))
        _assert_models_equivalent(model, loaded)

    def test_unknown_class_rejected(self, tmp_path):
        """A checkpoint naming an unregistered class fails with a clear error."""
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.Constant_ELT(),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=1.0,
            gamma_init=1.0,
            sigma_init=1.0,
        )
        checkpoint = model._to_checkpoint()
        checkpoint["eddy_lifetime_class"] = "DoesNotExist_ELT"
        path = tmp_path / "bad.pt"
        torch.save(checkpoint, path)

        with pytest.raises(ValueError, match="unknown model class"):
            stm.SpectralTensorModel.load_model(path)


class TestCalibrationProblemSaveModel:
    """CalibrationProblem.save_model records the calibration scaling factors."""

    def test_save_records_calibration_scales(self, tmp_path):
        """After a short calibration, the checkpoint carries freq/spectral scales."""
        torch.manual_seed(0)
        zref, ustar = 90.0, 0.556
        k1 = torch.logspace(-1, 2, 10, dtype=torch.float64) / zref
        data = generate_kaimal_spectra(k1=k1, zref=zref, ustar=ustar)

        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.Mann_ELT(),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=0.59 * zref,
            gamma_init=3.9,
            sigma_init=3.2 * ustar**2 / zref ** (2.0 / 3.0),
        )
        pb = CalibrationProblem(
            model=model,
            data=data,
            loss_params=drdmt.LossParameters(),
            output_directory=tmp_path,
        )
        pb.calibrate(optimizer_class=torch.optim.Adam, lr=1e-3, max_epochs=1)

        path = pb.save_model()
        assert path == tmp_path / "drd_model.pt"
        assert path.exists()

        loaded = stm.SpectralTensorModel.load_model(path)
        assert hasattr(loaded, "calibration_scales")
        assert loaded.calibration_scales["freq_scale"] == pytest.approx(float(pb.freq_scale))
        assert loaded.calibration_scales["spectral_scale"] == pytest.approx(float(pb.spectral_scale))
        assert np.isfinite(loaded.calibration_scales["freq_scale"])
        _assert_models_equivalent(model, loaded)

    def test_save_before_calibration_defaults_scales_to_one(self, tmp_path):
        """Saving an uncalibrated problem records identity scales rather than failing."""
        zref, ustar = 90.0, 0.556
        k1 = torch.logspace(-1, 2, 10, dtype=torch.float64) / zref
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.Constant_ELT(),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=1.0,
            gamma_init=1.0,
            sigma_init=1.0,
        )
        pb = CalibrationProblem(
            model=model,
            data=generate_kaimal_spectra(k1=k1, zref=zref, ustar=ustar),
            output_directory=tmp_path,
        )

        loaded = stm.SpectralTensorModel.load_model(pb.save_model(tmp_path / "uncalibrated"))
        assert loaded.calibration_scales == {"freq_scale": 1.0, "spectral_scale": 1.0}
