"""Tests for coherence loss integration."""

import polars as pl
import torch

from drdmannturb.parameters import IntegrationParameters, LossParameters
from drdmannturb.spectra_fitting.calibration import CalibrationProblem
from drdmannturb.spectra_fitting.loss_functions import LossAggregator
from drdmannturb.spectra_fitting.spectral_tensor_models import Mann_ELT, RDT_SpectralTensor, VonKarman_ESM


def _coherence_kwargs(value: float, length: int, device: str = "cpu"):
    tensor_val = torch.full((1, length), value, device=device)
    return {
        "coherence_data": {
            "model_u": tensor_val,
            "model_v": tensor_val,
            "model_w": tensor_val,
            "data_u": torch.zeros_like(tensor_val),
            "data_v": torch.zeros_like(tensor_val),
            "data_w": torch.zeros_like(tensor_val),
        }
    }


def test_coherence_loss_included(tmp_path):
    """Coherence contribution is added to total loss when enabled."""
    params = LossParameters(gamma_coherence=1.0)
    k_domain = torch.tensor([1.0, 2.0])
    agg = LossAggregator(params=params, ops_k_domain=k_domain, tb_log_dir=tmp_path)

    model_ops = torch.ones((1, k_domain.numel()))
    target_ops = model_ops.clone()

    loss = agg.eval(model_ops, target_ops, torch.zeros(1), epoch=0, **_coherence_kwargs(1.0, k_domain.numel()))

    # With unit coherence mismatch across three components, the coherence loss should be 1.0
    assert torch.isclose(loss, torch.tensor(1.0, device=loss.device))


def test_coherence_loss_skipped_when_disabled(tmp_path):
    """Coherence data is ignored if gamma_coherence is zero."""
    params = LossParameters(gamma_coherence=0.0)
    k_domain = torch.tensor([1.0, 2.0])
    agg = LossAggregator(params=params, ops_k_domain=k_domain, tb_log_dir=tmp_path)

    model_ops = torch.ones((1, k_domain.numel()))
    target_ops = model_ops.clone()

    loss = agg.eval(model_ops, target_ops, torch.zeros(1), epoch=0, **_coherence_kwargs(1.0, k_domain.numel()))

    assert torch.isclose(loss, torch.tensor(0.0, device=loss.device))


def test_long_form_coherence_grid_calibrates(tmp_path):
    """Repeated long-form coordinates are reduced to unique coherence axes."""
    ops = pl.DataFrame(
        {
            "freq": [0.1, 0.2],
            "uu": [1.0, 0.8],
            "vv": [0.9, 0.7],
            "ww": [0.6, 0.5],
            "uw": [float("nan"), float("nan")],
            "vw": [float("nan"), float("nan")],
            "uv": [float("nan"), float("nan")],
        }
    )
    coherence = pl.DataFrame(
        {
            "r": [2.0, 1.0, 2.0, 1.0],
            "freq": [0.2, 0.1, 0.1, 0.2],
            "coh_u": [0.4, 0.8, 0.7, 0.5],
            "coh_v": [0.3, 0.7, 0.6, 0.4],
            "coh_w": [0.2, 0.6, 0.5, 0.3],
        }
    )
    model = RDT_SpectralTensor(
        eddy_lifetime_model=Mann_ELT(),
        energy_spectrum_model=VonKarman_ESM(),
        L_init=1.0,
        gamma_init=1.0,
        sigma_init=1.0,
    )
    problem = CalibrationProblem(
        model=model,
        data={"ops": ops, "coherence": coherence},
        fit_coherence=True,
        loss_params=LossParameters(gamma_coherence=0.1),
        integration_params=IntegrationParameters(ops_num_points=1, coh_num_points=1),
        logging_directory=tmp_path,
    )

    problem.calibrate(optimizer_class=torch.optim.Adam, lr=1e-4, max_epochs=1)

    assert torch.isfinite(problem.loss)


def test_explicit_components_preserve_structural_zero_cross_spectra(tmp_path):
    """Unsupported nonzero cross targets can be excluded without changing model zeros."""
    ops = pl.DataFrame(
        {
            "freq": [0.1, 0.2],
            "uu": [1.0, 0.8],
            "vv": [0.9, 0.7],
            "ww": [0.6, 0.5],
            "uw": [-0.1, -0.08],
            "vw": [0.04, 0.03],
            "uv": [0.02, 0.01],
        }
    )
    model = RDT_SpectralTensor(
        eddy_lifetime_model=Mann_ELT(),
        energy_spectrum_model=VonKarman_ESM(),
        L_init=1.0,
        gamma_init=1.0,
        sigma_init=1.0,
    )
    problem = CalibrationProblem(
        model=model,
        data={"ops": ops, "coherence": None},
        fit_components=("uu", "vv", "ww", "uw"),
        integration_params=IntegrationParameters(ops_num_points=1),
        logging_directory=tmp_path,
    )

    problem.calibrate(optimizer_class=torch.optim.Adam, lr=1e-4, max_epochs=1)

    assert problem.fitted_components == ("uu", "vv", "ww", "uw")
    assert torch.isfinite(problem.loss)
    with torch.no_grad():
        model_ops = problem.OPS(torch.tensor([0.1, 0.2]) / problem.freq_scale)
    assert torch.equal(model_ops[4], torch.zeros_like(model_ops[4]))
    assert torch.allclose(model_ops[5], torch.zeros_like(model_ops[5]), rtol=0.0, atol=1e-16)
