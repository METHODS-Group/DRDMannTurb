"""Mann/RDT baseline regression tests.

These tests pin the current ``RDT_SpectralTensor`` implementation to the validated
v1.0.x implementation (``drdmannturb/spectra_fitting/power_spectra_rdt.py`` at commit
``fe3093b~1``, i.e. the JOSS-era code) and to the integrated one-point physics of the
Mann (1994) model. They are the guard rail for any future refactor (including the
JAX migration): if these fail, the baseline physics has drifted.
"""

import torch
from torch import Tensor

from drdmannturb.parameters import IntegrationParameters
from drdmannturb.spectra_fitting.one_point_spectra import OnePointSpectra
from drdmannturb.spectra_fitting.spectral_tensor_models import (
    Constant_ELT,
    Mann_ELT,
    RDT_SpectralTensor,
    VonKarman_ESM,
)


def _power_spectra_rdt_v10x(
    k: Tensor, beta: Tensor, E0: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Verbatim v1.0.x reference RDT formula (PowerSpectraRDT, commit fe3093b~1).

    Uses raw division, so it is only valid at generic wavevectors (k1 != 0, s != 0).
    Returns components in the v1.0.x order: Phi11, Phi22, Phi33, Phi13, Phi12, Phi23.
    """
    k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

    k30 = k3 + beta * k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    C2 = k2 * kk0 / torch.sqrt(s**3) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

    zeta1 = C1 - k2 / k1 * C2
    zeta2 = C1 * k2 / k1 + C2
    E0 = E0 / (4 * torch.pi)
    Phi11 = E0 / (kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
    Phi22 = E0 / (kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
    Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
    Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

    Phi12 = E0 / (kk0**2) * (-k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2)
    Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

    return Phi11, Phi22, Phi33, Phi13, Phi12, Phi23


class TestV10xReferenceParity:
    """Current RDT tensor must reproduce the v1.0.x formula at generic wavevectors."""

    def test_rdt_matches_v10x_reference(self):
        """Compare RDT_SpectralTensor against the embedded v1.0.x formula.

        NOTE the component-order change across the v1.1 refactor:
        v1.0.x returned (..., Phi13, Phi12, Phi23); the current model returns
        (..., Phi13, Phi23, Phi12) = (uu, vv, ww, uw, vw, uv).
        """
        L_val, gamma_val, sigma_val = 10.0, 4.0, 3.0

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, L_val, gamma_val, sigma_val)

        # Generic wavevectors only (reference formula uses raw division)
        torch.manual_seed(0)
        k = torch.randn(200, 3, dtype=torch.float64)
        k = k[(k[:, 0].abs() > 1e-3) & (k[:, :2].norm(dim=-1) > 1e-3)]

        L = torch.exp(model.log_L)
        gamma = torch.exp(model.log_gamma)
        sigma = torch.exp(model.log_sigma)

        # Reproduce the model's own beta/E0 plumbing
        beta = elt(k, L, gamma)
        k0 = k.clone()
        k0[..., 2] = k[..., 2] + beta * k[..., 0]
        E0 = sigma * L ** (5.0 / 3.0) * esm(k0, L)

        ref = _power_spectra_rdt_v10x(k, beta, E0)
        out = model(k)

        # Map current order (uu, vv, ww, uw, vw, uv) onto reference order
        ref_phi11, ref_phi22, ref_phi33, ref_phi13, ref_phi12, ref_phi23 = ref
        expected = [ref_phi11, ref_phi22, ref_phi33, ref_phi13, ref_phi23, ref_phi12]

        for i, (current, reference) in enumerate(zip(out, expected)):
            torch.testing.assert_close(
                current.to(torch.float64),
                reference.to(torch.float64),
                rtol=1e-10,
                atol=1e-12,
                msg=f"Component {i} diverges from v1.0.x reference",
            )


class TestMannOnePointPhysics:
    """Integrated one-point spectra must satisfy the known Mann-model physics."""

    def _make_ops(self, elt) -> OnePointSpectra:
        model = RDT_SpectralTensor(elt, VonKarman_ESM(), 0.59, 3.9, 3.2)
        return OnePointSpectra(
            spectral_tensor_model=model,
            integration_params=IntegrationParameters(),
        )

    def test_one_point_cross_spectra_structure(self):
        """F11, F22, F33 > 0; F13 < 0 under shear; F23 = F12 = 0 by k2-oddness.

        The 3D components Phi23, Phi12 are NOT zero pointwise; only their k2/k3
        integrals vanish. This distinction is exactly what the removed
        anti-symmetrization hack got wrong.
        """
        ops = self._make_ops(Mann_ELT())
        k1 = torch.logspace(-2, 2, 8)

        with torch.no_grad():
            kF = ops(k1)

        F_uu, F_vv, F_ww, F_uw, F_vw, F_uv = kF

        assert torch.all(F_uu > 0)
        assert torch.all(F_vv > 0)
        assert torch.all(F_ww > 0)

        # Shear (gamma > 0) produces strictly negative uw momentum flux
        assert torch.all(F_uw < 0)

        # vw and uv vanish upon integration (odd in k2); compare against the
        # magnitude of the auto-spectra at the same k1
        scale = F_uu.abs()
        assert torch.all(F_vw.abs() / scale < 1e-6)
        assert torch.all(F_uv.abs() / scale < 1e-6)

    def test_isotropic_limit_no_uw_flux(self):
        """With gamma ~ 0 (no shear distortion), the uw cross-spectrum vanishes."""
        model = RDT_SpectralTensor(Constant_ELT(), VonKarman_ESM(), 0.59, 1e-12, 3.2)
        ops = OnePointSpectra(spectral_tensor_model=model, integration_params=IntegrationParameters())

        k1 = torch.logspace(-1, 1, 5)
        with torch.no_grad():
            kF = ops(k1)

        F_uu, F_uw = kF[0], kF[3]
        assert torch.all(F_uu > 0)
        assert torch.all(F_uw.abs() / F_uu < 1e-8)
