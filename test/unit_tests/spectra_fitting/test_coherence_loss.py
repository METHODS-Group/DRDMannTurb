"""Tests for coherence loss integration."""

import torch

from drdmannturb.parameters import LossParameters
from drdmannturb.spectra_fitting.loss_functions import LossAggregator


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
