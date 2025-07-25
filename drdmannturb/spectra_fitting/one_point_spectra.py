"""
Several integrators for the spectral tensor models.

Several physical quantities of interest in spectral turbulence models are calculated
as certain integrals of the spectral tensor. This module implements some of these.
"""

import torch
import torch.nn as nn

from ..parameters import IntegrationParameters
from .spectral_tensor_models import SpectralTensorModel


class OnePointSpectra(nn.Module):
    """Calculates the one-point spectra of a provided spectral tensor model."""

    spectral_tensor_model: SpectralTensorModel

    use_coherence: bool

    def __init__(
        self,
        spectral_tensor_model: SpectralTensorModel,
        integration_params: IntegrationParameters | None = None,
        use_coherence: bool = False,
    ):
        """Initialize the one-point spectra calculator."""
        super().__init__()

        # Class attributes
        self.spectral_tensor_model = spectral_tensor_model
        self.use_coherence = use_coherence

        if integration_params is None:
            integration_params = IntegrationParameters()

        ####
        # OPS grid
        # k2 grid
        p1, p2, N = integration_params.ops_log_min, integration_params.ops_log_max, integration_params.ops_num_points
        grid_zero = torch.tensor([0])
        grid_plus = torch.logspace(p1, p2, N)
        grid_minus = -torch.flip(grid_plus, dims=[0])

        ops_grid_OneDim = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.ops_grid_k2 = ops_grid_OneDim.clone()
        self.ops_grid_k3 = ops_grid_OneDim.clone()

        self.ops_meshgrid23 = torch.meshgrid(self.ops_grid_k2, self.ops_grid_k3, indexing="ij")

        ####
        # Coherence grid
        # TODO: Reimplement the coherence branching
        p1, p2 = integration_params.coh_log_min, integration_params.coh_log_max
        N_coh = integration_params.coh_num_points
        grid_zero_coh = torch.tensor([0])
        grid_plus_coh = torch.logspace(p1, p2, N_coh)
        grid_minus_coh = -torch.flip(grid_plus_coh, dims=[0])

        self.coh_grid_k2 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach()
        self.coh_grid_k3 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach()

        self.coh_meshgrid23 = torch.meshgrid(self.coh_grid_k2, self.coh_grid_k3, indexing="ij")

    def forward(self, k1_input: torch.Tensor) -> torch.Tensor:
        """Evaluate the frequency-weighted one-point spectra."""
        k = torch.stack(torch.meshgrid(k1_input, self.ops_grid_k2, self.ops_grid_k3, indexing="ij"), dim=-1)

        # Evaluate the spectral tensor model
        phi_arr = self.spectral_tensor_model(k)

        assert len(phi_arr) == 6, "Something is wrong with the spectral tensor model output."

        # Calculate the frequency-weighted one-point spectra
        kF = torch.stack([k1_input * self.quad23(phi_i) for phi_i in phi_arr])

        return kF

    @torch.jit.export
    def quad23(self, f: torch.Tensor) -> torch.Tensor:
        """Evaluate the 23-point quadrature."""
        # Integration over k3
        quad = torch.trapz(f, x=self.ops_grid_k3, dim=-1)
        # Integration over k2, fix k3 = 0 since slices are identical in meshgrid
        quad = torch.trapz(quad, x=self.ops_grid_k2, dim=-1)

        return quad

    def spectral_coherence(
        self,
        k1_input: torch.Tensor,
        spatial_separations: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the spectral coherence in the cross-stream direction."""
        raise NotImplementedError("Not yet implemented.")

    @torch.jit.export
    def get_div(self, Phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the divergence of an evaluated spectral tensor model."""
        raise NotImplementedError("Not yet implemented.")
