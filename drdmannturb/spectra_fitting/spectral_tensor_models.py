"""Physical models composing the DRD models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import hyp2f1  # Just for the Mann ELT

from ..nn_modules import TauNet

###############################################################################
# Eddy Lifetime Functions
###############################################################################


class EddyLifetimeModel(nn.Module):
    """Base class for eddy lifetime models."""

    def __init__(self):
        super().__init__()

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Compute eddy lifetime tau(k)."""
        raise NotImplementedError("Subclasses must implement the forward pass.")

    def _config(self) -> dict:
        """Return constructor arguments needed to rebuild this instance (learned values live in state_dict)."""
        return {}

    @classmethod
    def _from_config(cls, config: dict) -> "EddyLifetimeModel":
        """Reconstruct an instance from a ``_config()`` dictionary."""
        return cls(**config)


class TauNet_ELT(EddyLifetimeModel):
    """The DRD eddy lifetime model."""

    tauNet: TauNet

    def __init__(self, taunet: TauNet):
        super().__init__()
        self.tauNet = taunet

    def _config(self) -> dict:
        """Return the TauNet architecture needed to rebuild this instance."""
        return {
            "n_layers": self.tauNet.n_layers,
            "hidden_layer_sizes": list(self.tauNet.hidden_layer_sizes),
            "activations": [type(act).__name__ for act in self.tauNet.activations],
            "learn_nu": self.tauNet.Ra.fg_learn_nu,
            "nu_init": float(self.tauNet.Ra.nu),
        }

    @classmethod
    def _from_config(cls, config: dict) -> "TauNet_ELT":
        """Reconstruct a TauNet_ELT from a ``_config()`` dictionary."""
        config = dict(config)
        activation_names = config.pop("activations")
        try:
            activations = [getattr(nn, name)() for name in activation_names]
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f"Cannot reconstruct activations {activation_names}; only argument-free torch.nn activations "
                "are supported by save_model/load_model."
            ) from e
        return cls(taunet=TauNet(activations=activations, **config))

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the DRD eddy lifetime model."""
        # NOTE: TauNet takes in the entire k grid, unlike the other models.
        kL = L * k

        return gamma * self.tauNet(kL)


class Mann_ELT(EddyLifetimeModel):
    """The Mann eddy lifetime model.

    .. warning::
        This model is evaluated through SciPy's ``hyp2f1`` (there is no torch implementation
        of the Gauss hypergeometric function), so it is **not differentiable** with respect to
        ``L`` and is evaluated on the CPU: gradients do not flow through the eddy lifetime
        here. Use ``TauNet_ELT`` or ``TwoThirds_ELT`` when a differentiable lifetime is needed.
    """

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the Mann eddy lifetime model."""
        # Move inputs to CPU and then to numpy for SciPy compatibility
        k_norm_np = k.norm(dim=-1).detach().cpu().numpy()
        L_val = float(L.detach().cpu())
        kL = L_val * k_norm_np

        y = np.zeros_like(kL)
        mask = kL > 0.0
        if np.any(mask):
            t = kL[mask]
            y[mask] = (t ** (-2.0 / 3.0)) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(t ** (-2.0))))

        tau = torch.tensor(y, dtype=k.dtype, device=k.device)
        return gamma * tau


class TwoThirds_ELT(EddyLifetimeModel):
    """The two-thirds eddy lifetime model."""

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the two-thirds eddy lifetime model."""
        kL = L * k.norm(dim=-1)
        tau = torch.where(kL > 0.0, kL ** (-2.0 / 3.0), torch.zeros_like(kL))

        return gamma * tau


class Constant_ELT(EddyLifetimeModel):
    """The constant eddy lifetime model."""

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the eddy lifetime model."""
        kL = L * k.norm(dim=-1)

        tau = torch.ones_like(kL)

        return gamma * tau


###############################################################################
# Energy Spectrum Functions
###############################################################################


class EnergySpectrumModel(nn.Module):
    """Base class for energy spectrum models."""

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the energy spectrum model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")

    def _config(self) -> dict:
        """Return constructor arguments needed to rebuild this instance (learned values live in state_dict)."""
        return {}

    @classmethod
    def _from_config(cls, config: dict) -> "EnergySpectrumModel":
        """Reconstruct an instance from a ``_config()`` dictionary."""
        return cls(**config)


class VonKarman_ESM(EnergySpectrumModel):
    """The standard Von Karman energy spectrum model.

    Note that this only includes scaling by the parameter L.
    """

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the Von Karman energy spectrum."""
        # Compute k = |k|
        k_norm = k.norm(dim=-1)
        kL = k_norm * L

        # Guarded evaluation: E(0) = 0, with k_norm replaced by 1 under the mask
        # so the k^{-5/3} factor never produces inf/nan to propagate.
        mask = k_norm > 0.0
        kn_safe = torch.where(mask, k_norm, torch.ones_like(k_norm))
        parenthetical = kL / torch.sqrt(1.0 + kL**2)
        E = (kn_safe ** (-5.0 / 3.0)) * (parenthetical ** (17.0 / 3.0))

        return torch.where(mask, E, torch.zeros_like(E))


class Learnable_ESM(EnergySpectrumModel):
    r"""Learnable energy spectrum model.

    A generalization of the Von Karman energy spectrum with learnable shape exponents,

    .. math::
        E(k) \propto \frac{(kL)^p}{\left(1 + (kL)^2\right)^q},

    so that :math:`E(k) \sim k^p` in the energy-containing range :math:`kL \ll 1` and
    :math:`E(k) \sim k^{p - 2q}` in the inertial range :math:`kL \gg 1`. Positivity of
    :math:`p` and :math:`q` is enforced through a softplus reparametrization.

    With the default initial values :math:`p = 4` and :math:`q = 17/6`, this reproduces
    the Von Karman spectrum shape exactly (including the :math:`k^{-5/3}` inertial-range
    decay); the constant factor :math:`L^{5/3}` separating it from ``VonKarman_ESM`` is
    absorbed by the spectral tensor model's learnable amplitude ``sigma``.
    """

    def __init__(
        self,
        p_init: float = 4.0,
        q_init: float = 17.0 / 6.0,
    ):
        super().__init__()

        self._raw_p = nn.Parameter(torch.tensor(np.log(np.exp(p_init) - 1.0)))
        self._raw_q = nn.Parameter(torch.tensor(np.log(np.exp(q_init) - 1.0)))

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(raw)

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the learnable energy spectrum model."""
        k_norm = k.norm(dim=-1)
        kL = L * k_norm
        p = self._positive(self._raw_p)
        q = self._positive(self._raw_q)

        return (kL**p) / ((1.0 + kL**2) ** q)


###############################################################################
# Spectral Tensor Models
###############################################################################


class SpectralTensorModel(nn.Module):
    """Base class for spectral tensor models."""

    eddy_lifetime_model: EddyLifetimeModel
    energy_spectrum_model: EnergySpectrumModel

    log_L: nn.Parameter
    log_gamma: nn.Parameter
    log_sigma: nn.Parameter

    def __init__(
        self,
        eddy_lifetime_model: EddyLifetimeModel,
        energy_spectrum_model: EnergySpectrumModel,
        L_init: float,
        gamma_init: float,
        sigma_init: float,
    ):
        super().__init__()
        self.eddy_lifetime_model = eddy_lifetime_model
        self.energy_spectrum_model = energy_spectrum_model

        # Check that initial values are positive
        if L_init <= 0.0:
            raise ValueError("L_init must be positive.")
        if gamma_init <= 0.0:
            raise ValueError("gamma_init must be positive.")
        if sigma_init <= 0.0:
            raise ValueError("sigma_init must be positive.")

        # Learnable scaling parameters
        # NOTE: These are stored as log-transformed values to cleanly ensure positivity
        self.log_L = nn.Parameter(torch.tensor(np.log(L_init), dtype=torch.get_default_dtype()))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.get_default_dtype()))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.get_default_dtype()))

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the spectral tensor model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")

    def _to_checkpoint(self) -> dict:
        """Build a serializable checkpoint dictionary for this model."""
        return {
            "format_version": 1,
            "model_class": type(self).__name__,
            "eddy_lifetime_class": type(self.eddy_lifetime_model).__name__,
            "eddy_lifetime_config": self.eddy_lifetime_model._config(),
            "energy_spectrum_class": type(self.energy_spectrum_model).__name__,
            "energy_spectrum_config": self.energy_spectrum_model._config(),
            "state_dict": self.state_dict(),
        }

    def save_model(self, save_path: str | Path) -> Path:
        """Save the model to a ``.pt`` checkpoint file.

        The checkpoint stores the model's ``state_dict`` along with the class names and
        constructor configuration of its eddy lifetime and energy spectrum models, so
        that :py:meth:`load_model` can rebuild it without access to the original code
        that constructed it.

        Parameters
        ----------
        save_path : str | Path
            File to write; a ``.pt`` suffix is appended if no suffix is given. Parent
            directories are created as needed.

        Returns
        -------
        Path
            The path the checkpoint was written to.
        """
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._to_checkpoint(), save_path)
        return save_path

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "SpectralTensorModel":
        """Reconstruct a model from a checkpoint dictionary."""
        try:
            model_cls = _SPECTRAL_TENSOR_MODELS[checkpoint["model_class"]]
            elt_cls = _EDDY_LIFETIME_MODELS[checkpoint["eddy_lifetime_class"]]
            esm_cls = _ENERGY_SPECTRUM_MODELS[checkpoint["energy_spectrum_class"]]
        except KeyError as e:
            raise ValueError(f"Checkpoint references an unknown model class: {e}") from e

        model = model_cls(
            eddy_lifetime_model=elt_cls._from_config(checkpoint["eddy_lifetime_config"]),
            energy_spectrum_model=esm_cls._from_config(checkpoint["energy_spectrum_config"]),
            L_init=1.0,  # Placeholder; overwritten by the state_dict below.
            gamma_init=1.0,
            sigma_init=1.0,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @classmethod
    def load_model(cls, load_file: str | Path) -> "SpectralTensorModel":
        """Load a model from a checkpoint file written by :py:meth:`save_model`.

        Parameters
        ----------
        load_file : str | Path
            Path to the checkpoint file.

        Returns
        -------
        SpectralTensorModel
            The reconstructed model, with all learned parameters restored. If the
            checkpoint was written by ``CalibrationProblem.save_model``, the calibration
            scaling factors are attached as a ``calibration_scales`` attribute
            (a dict with keys ``freq_scale`` and ``spectral_scale``).
        """
        checkpoint = torch.load(load_file, map_location="cpu", weights_only=True)
        model = cls._from_checkpoint(checkpoint)
        if "calibration" in checkpoint:
            model.calibration_scales = checkpoint["calibration"]
        return model


class RDT_SpectralTensor(SpectralTensorModel):
    """The Rapid Distortion Theory spectral tensor model."""

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        mask = denom != 0
        denom_safe = torch.where(mask, denom, torch.ones_like(denom))
        return (numer / denom_safe) * mask.to(numer.dtype)

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the RDT spectral tensor model."""
        # Calculate the scaling parameters
        L = torch.exp(self.log_L)
        gamma = torch.exp(self.log_gamma)
        sigma = torch.exp(self.log_sigma)

        # NOTE: The following was previously in OnePointSpectra.forward()
        beta = self.eddy_lifetime_model(k, L, gamma)

        k0 = k.clone()
        k0[..., 2] = k[..., 2] + beta * k[..., 0]

        # Calculate energy spectrum for "Phi_VK"
        energy_spectrum = self.energy_spectrum_model(k0, L)
        E0 = sigma * L ** (5.0 / 3.0) * energy_spectrum

        # Split k into components
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

        # Calculate
        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        C1 = self._safe_div(beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30), kk * s)
        C2 = self._safe_div(k2 * kk0, torch.sqrt(s**3)) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

        inv_k1 = self._safe_div(torch.ones_like(k1), k1)
        zeta1 = C1 - k2 * inv_k1 * C2
        zeta2 = C1 * k2 * inv_k1 + C2

        E0 = E0 / (4 * torch.pi)

        # Spectral tensor components with guarded division
        Phi11 = self._safe_div(E0, kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = self._safe_div(E0, kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = self._safe_div(E0, kk**2) * (k1**2 + k2**2)
        Phi13 = self._safe_div(E0, kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)
        Phi12 = self._safe_div(E0, kk0**2) * (
            -k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2
        )
        Phi23 = self._safe_div(E0, kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # In order, uu, vv, ww, uw, vw, uv
        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12


###############################################################################
# Checkpoint registries
###############################################################################

# Maps class names stored in checkpoints back to classes, for save_model/load_model.
_EDDY_LIFETIME_MODELS: dict[str, type[EddyLifetimeModel]] = {
    cls.__name__: cls for cls in (TauNet_ELT, Mann_ELT, TwoThirds_ELT, Constant_ELT)
}
_ENERGY_SPECTRUM_MODELS: dict[str, type[EnergySpectrumModel]] = {
    cls.__name__: cls for cls in (VonKarman_ESM, Learnable_ESM)
}
_SPECTRAL_TENSOR_MODELS: dict[str, type[SpectralTensorModel]] = {cls.__name__: cls for cls in (RDT_SpectralTensor,)}
