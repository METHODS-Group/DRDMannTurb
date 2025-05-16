"""Calibration of the eddy lifetime function and spectral tensors."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# TODO: Numpy and Scipy are required for the Mann eddy lifetime function and are not JIT'able, GPU-compatible,
#       or autodifferentiable.
import numpy as np
import optax
from scipy.special import hyp2f1

#############################################################################################################
# Define Eddy Lifetime functions
#############################################################################################################

#################################################
# Rational kernel
class RationalKernel(eqx.Module):
    r"""Construct the rational function kernel.

    In order to meet asymptotic behavior requirements for the eddy lifetime function,
    we compose a neural network with this kernel, which implements

    .. math::
        \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
        {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}},
        \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k})

    Args:
        nu_value: Initial value for nu (default: -1/3)
        learn_nu: Whether to make nu a learnable parameter (default: True)
    """

    nu: jnp.ndarray
    learn_nu: bool

    def __init__(self, nu_value: float = -1./3., learn_nu: bool = True, *, key: Optional[jax.random.PRNGKey] = None):
        """Initialize the Rational kernel module."""
        r"""
        NOTE: nu appears in the following expression:

        tau(k) =      T |a|^{\nu - 2/3}
                     --------------------
                     (1 + |a|^2)^{\nu / 2}

        and is meant to guarantee the following asymptotic behavior:

        tau^IEC (k) = { k^{-1}   for k \to 0
                      { k^{-3/2} for k \to \infty
        """
        self.learn_nu = learn_nu
        if learn_nu:
            self.nu = jnp.array(nu_value)
        else:
            self.nu = eqx.static_field(jnp.array(nu_value))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward method implementation.

        Parameters
        ----------
        x : jnp.ndarray
            Network input

        Returns
        -------
        jnp.ndarray
            Network output
        """
        a = self.nu - (2. / 3.)
        b = self.nu

        out = jnp.abs(x)
        out = jnp.power(out, a) / jnp.power(1. + jnp.power(out, 2.), b / 2.)

        return out

#################################################
# TauNet
class TauNet(eqx.Module):
    r"""The neural network approximation of the eddy lifetime function.

    This is a configurable multi-layer perceptron composed with the rational kernel, required
    to guarantee the asymptotic behavior of the eddy lifetime function.
    """

    # MLP layers
    layers: list[eqx.nn.Linear]
    activations: list[Callable]

    # Rational kernel
    kernel: RationalKernel

    def __init__(
        self,
        hidden_layer_sizes: Sequence[Tuple[int, Callable]] = [(10, jax.nn.relu), (10, jax.nn.relu), (10, jax.nn.relu)],
        learn_nu: bool = True,
        nu_value: float = -1./3.,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Initialize the TauNet.

        Args:
            hidden_layer_sizes: Sequence of (layer_size, activation_function) tuples
            learn_nu: Whether to make nu a learnable parameter
            nu_value: Initial value for nu
            key: PRNG key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Extract layer sizes and activations from the sequence
        layer_sizes = [size for size, _ in hidden_layer_sizes]
        self.activations = [act for _, act in hidden_layer_sizes]

        # Initialize MLP layers
        keys = jax.random.split(key, len(layer_sizes) + 2)  # +2 for input and output layers
        self.layers = []

        # Input layer (3 -> first hidden)
        self.layers.append(eqx.nn.Linear(3, layer_sizes[0], use_bias=False, key=keys[0]))

        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], use_bias=False, key=keys[i + 1])
            )

        # Output layer (last hidden -> 3)
        self.layers.append(eqx.nn.Linear(layer_sizes[-1], 3, use_bias=False, key=keys[-1]))

        # Initialize rational kernel
        self.kernel = RationalKernel(nu_value=nu_value, learn_nu=learn_nu)

        # Add small noise to weights for better initialization
        noise_magnitude = 1.0e-9
        for i, layer in enumerate(self.layers):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, layer.weight.shape) * noise_magnitude
            # Use equinox.tree_at to create a new Linear layer with perturbed weights
            self.layers[i] = eqx.tree_at(
                lambda ell: ell.weight,
                layer,
                layer.weight + noise,
            )

    def __call__(self, k: jnp.ndarray) -> jnp.ndarray:
        """Compute the learned eddy lifetime function.

        Parameters
        ----------
        k : jnp.ndarray
            Input wavevector (shape: [..., 3])

        Returns
        -------
        jnp.ndarray
            Eddy lifetime values
        """
        # Compute |k|
        k_abs = jnp.abs(k)

        # Pass through MLP
        x = k_abs
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)

        # Compute norm of MLP output
        k_mod = jnp.linalg.norm(x, axis=-1)

        # Apply rational kernel
        tau = self.kernel(k_mod)

        return tau


#############################################################################################################
# Spectral Tensor definitions
#############################################################################################################

def VKEnergySpectrum(kL: jnp.ndarray) -> jnp.ndarray:
    """Von Karman energy spectrum (dimensionless, no scaling).

    Parameters
    ----------
    kL : jnp.ndarray
        Non-dimensional wave-number (k * L)

    Returns
    -------
    jnp.ndarray
        Spectrum values evaluated at *kL*.
    """
    return kL ** 4 / (1.0 + kL ** 2) ** (17.0 / 6.0)


# NOTE: The hypergeometric 2F1 required for the analytical Mann eddy-lifetime is not available in JAX.
#       We therefore keep a NumPy/SciPy helper.  It is *not* used inside jitted code paths.

def MannEddyLifetime(kL: np.ndarray) -> np.ndarray:
    """Reference implementation of Mann's eddy-lifetime (CPU only).

    This is retained for comparison / initialisation but is *not* part of the JAX computational graph.
    """
    x = np.asarray(kL)
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -x ** (-2)))
    return y


def PowerSpectraRDT(k: jnp.ndarray, beta: jnp.ndarray, E0: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Classical Rapid-Distortion (RDT) spectral tensor.

    Parameters
    ----------
    k    : (..., 3) jnp.ndarray
        Wave-vector array.
    beta : (...) jnp.ndarray
        Distortion parameter \(\beta = \Gamma\,\tau(k)\).
    E0   : (...) jnp.ndarray
        Von-Karman spectrum evaluated at the distorted wave-vector *k0* (already scaled).

    Returns
    -------
    tuple of jnp.ndarray
        (Phi11, Phi22, Phi33, Phi13, Phi12, Phi23) – each with shape matching *beta*.
    """
    k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

    k30 = k3 + beta * k1
    kk0 = k1 ** 2 + k2 ** 2 + k30 ** 2
    kk = k1 ** 2 + k2 ** 2 + k3 ** 2
    s = k1 ** 2 + k2 ** 2

    # Avoid division by zero in *s*
    s_safe = jnp.where(s == 0, 1.0e-12, s)

    C1 = beta * k1 ** 2 * (kk0 - 2 * k30 ** 2 + beta * k1 * k30) / (kk * s_safe)
    C2 = (
        k2
        * kk0
        / jnp.sqrt(s_safe ** 3)
        * jnp.arctan2(beta * k1 * jnp.sqrt(s_safe), kk0 - k30 * k1 * beta)
    )

    k1_safe = jnp.where(k1 == 0, 1.0e-12, k1)

    zeta1 = C1 - k2 / k1_safe * C2
    zeta2 = C1 * k2 / k1_safe + C2

    E0_scaled = E0 / (4.0 * jnp.pi)

    Phi11 = E0_scaled / (kk0 ** 2) * (
        kk0 - k1 ** 2 - 2 * k1 * k30 * zeta1 + (k1 ** 2 + k2 ** 2) * zeta1 ** 2
    )
    Phi22 = E0_scaled / (kk0 ** 2) * (
        kk0 - k2 ** 2 - 2 * k2 * k30 * zeta2 + (k1 ** 2 + k2 ** 2) * zeta2 ** 2
    )
    Phi33 = E0_scaled / (kk ** 2) * (k1 ** 2 + k2 ** 2)
    Phi13 = E0_scaled / (kk * kk0) * (-k1 * k30 + (k1 ** 2 + k2 ** 2) * zeta1)

    Phi12 = E0_scaled / (kk0 ** 2) * (
        -k1 * k2
        - k1 * k30 * zeta2
        - k2 * k30 * zeta1
        + (k1 ** 2 + k2 ** 2) * zeta1 * zeta2
    )
    Phi23 = E0_scaled / (kk * kk0) * (
        -k2 * k30 + (k1 ** 2 + k2 ** 2) * zeta2
    )

    return Phi11, Phi22, Phi33, Phi13, Phi12, Phi23

# Define the RDT spectral tensor

class RDT_spectral_tensor(eqx.Module):
    """Implements the RDT spectral tensor model."""

    eddy_lifetime: TauNet

    L: jnp.ndarray  # Length scale
    Gamma: jnp.ndarray  # Characteristic time scale
    sigma: jnp.ndarray  # Spectrum amplitude magnitude

    # Grids for k2 and k3 - these define the dimensions for spectral computation
    k2_grid: jnp.ndarray = eqx.static_field()
    k3_grid: jnp.ndarray = eqx.static_field()


    def __init__(
        self,
        eddy_lifetime: TauNet,
        L_init: float = 1.0,
        Gamma_init: float = 1.0,
        sigma_init: float = 1.0,
        k2_k3_params: Optional['k2_k3_parameters'] = None
    ):

        self.eddy_lifetime = eddy_lifetime

        # Initialize learnable scalar parameters
        # In Equinox, simple JAX arrays are traceable and thus learnable by default if part of the model's tree.
        self.L = jnp.array(L_init)
        self.Gamma = jnp.array(Gamma_init)
        self.sigma = jnp.array(sigma_init)

        # Initialize grids
        if k2_k3_params is None:
            k2_k3_params = k2_k3_parameters() # Use default if not provided

        self.k2_grid = jnp.logspace(
            k2_k3_params.k2_min_p,
            k2_k3_params.k2_max_p,
            k2_k3_params.k2_points
        )
        self.k3_grid = jnp.logspace(
            k2_k3_params.k3_min_p,
            k2_k3_params.k3_max_p,
            k2_k3_params.k3_points
        )

    def __call__(self, k1_input: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """
        Calculates the spectral tensor components Phi_ij(k1, k2, k3).
        The output shape will be (num_components, len(k1_input), len(k2_grid), len(k3_grid)).
        """
        # Create the full 3D wavevector grid
        # k1_input: [Nk1]
        # self.k2_grid: [Nk2]
        # self.k3_grid: [Nk3]
        # We need k_tensor of shape [Nk1, Nk2, Nk3, 3]

        K1, K2, K3 = jnp.meshgrid(k1_input, self.k2_grid, self.k3_grid, indexing='ij')
        # K1, K2, K3 will each have shape [Nk1, Nk2, Nk3]

        k_tensor = jnp.stack([K1, K2, K3], axis=-1) # Shape [Nk1, Nk2, Nk3, 3]

        # Calculate beta (eddy lifetime effect)
        # self.eddy_lifetime expects k of shape [..., 3] and returns tau of shape [...]
        # The original PyTorch code: self.beta = self.EddyLifetime()
        # Original EddyLifetime: self.TimeScale * tau
        #   tau depends on self.LengthScale * k.norm(dim=-1)
        #   or self.tauNet(k*self.LengthScale)

        # Let's follow the tauNet path from PyTorch OnePointSpectra:
        # tau = tau0 + self.tauNet(k*self.LengthScale)
        # beta = self.TimeScale * tau
        # For now, let's assume eddy_lifetime (TauNet) directly gives the 'tau' part
        # and we apply scales L and Gamma (TimeScale).

        # k_scaled_for_tauNet = k_tensor * self.L # Scale k by length scale for tauNet input
        # The TauNet in JAX_calibration.py takes k (not k*L) and computes tau.
        # Its __call__ method takes k (wavevector) and returns tau.
        # The original PyTorch OnePointSpectra.EddyLifetime has:
        #   kL = self.LengthScale * k.norm(dim=-1)
        #   tau_nn_input = k * self.LengthScale (for tauNet type)
        #   tau_output = self.tauNet(tau_nn_input)
        #   final_beta = self.TimeScale * (initial_guess_tau + tau_output)
        # JAX TauNet computes tau from k. So we need to decide if L scaling is part of TauNet's job or external.
        # Given TauNet definition, it seems to operate on 'k' directly.
        # The 'beta' in PowerSpectraRDT is 'gamma * tau(k*L)' in some formulations,
        # or directly 'tau_effective(k)'.
        # Let's assume self.eddy_lifetime(k_tensor) gives the core dimensionless lifetime.

        # The input `k` to `TauNet` should be the physical wavevector.
        # `TauNet` then learns the lifetime function `tau(k)`.
        # This `tau(k)` is then scaled by `Gamma` (Characteristic time scale)
        # The `L` (Length scale) affects `k0L` for `VKEnergySpectrum`.

        # Compute dimensionless eddy lifetime tau using the neural network.
        # Reshape to (N_total, 3) because eqx.nn.Linear expects a trailing feature dim only.
        k_flat = k_tensor.reshape((-1, 3))  # (Nk1*Nk2*Nk3, 3)
        tau_flat = jax.vmap(self.eddy_lifetime)(k_flat)  # vectorised
        tau_values = tau_flat.reshape(k_tensor.shape[:-1])  # (Nk1, Nk2, Nk3)

        # This tau_values is equivalent to beta/Gamma or the "shape" of the eddy lifetime.
        # The 'beta' in RDT equations is often 'Gamma * tau_characteristic_function(kL)'
        # In the PyTorch code, `beta = self.EddyLifetime()`, and
        # `EddyLifetime` returns `self.TimeScale * tau_calc`.
        # `tau_calc` can come from `self.tauNet(k*self.LengthScale)` if type is `tauNet`.
        # This implies the TauNet itself might be trained to work with k*L.
        # However, the JAX `TauNet` takes `k` directly.
        # If `TauNet` output is `tau(k_physical)`, then `beta = self.Gamma * tau(k_physical)`.

        beta = self.Gamma * tau_values # Shape [Nk1, Nk2, Nk3]

        # Calculate k0 and k0L for Von Karman energy spectrum
        # k0 = k; k0[...,2] = k[...,2] + beta * k[...,0]
        # This definition of k0 seems to use 'beta' which is 'Gamma * tau'.
        # This matches the structure `k3_0 = k3 + Gamma * tau * k1`
        k0 = k_tensor.at[..., 2].add(beta * k_tensor[..., 0]) # Shape [Nk1, Nk2, Nk3, 3]

        # k0L for VKEnergySpectrum: L * ||k0||
        k0_norm = jnp.linalg.norm(k0, axis=-1) # Shape [Nk1, Nk2, Nk3]
        k0L = self.L * k0_norm # Shape [Nk1, Nk2, Nk3]

        # Calculate E0 (Von Karman energy spectrum)
        # VKEnergySpectrum expects kL
        E0 = self.sigma * VKEnergySpectrum(k0L) # Shape [Nk1, Nk2, Nk3]

        # Calculate spectral tensor components Phi_ij using PowerSpectraRDT
        # PowerSpectraRDT(k, beta, E0)
        # k should be [..., 3], beta [...], E0 [...]
        # It returns a tuple of Phi_ij, each of shape [...] matching E0.
        # So, Phi_ij will have shape [Nk1, Nk2, Nk3]
        # The k_tensor is already [Nk1, Nk2, Nk3, 3]
        # beta is [Nk1, Nk2, Nk3]
        # E0 is [Nk1, Nk2, Nk3]
        # PowerSpectraRDT needs k, beta, E0 broadcastable. We might need to unsqueeze beta and E0.
        # k1, k2, k3 = k[...,0], k[...,1], k[...,2]
        # beta and E0 are used in scalar ways with components of k.

        # PowerSpectraRDT will operate on the last dimension of k_tensor.
        # beta and E0 need to be compatible. They are [Nk1, Nk2, Nk3]. This is fine.

        # The function PowerSpectraRDT takes k[...,0],k[...,1],k[...,2], beta, E0
        # k is (...,3), beta is (...), E0 is (...)
        # output is (Phi11, Phi22, Phi33, Phi13, Phi12, Phi23) where each Phi_ij is (...)
        phi_components = PowerSpectraRDT(k_tensor, beta, E0) # tuple of 6 arrays, each [Nk1, Nk2, Nk3]

        # We need to return these components in a way that OnePointSpectra can integrate.
        # OnePointSpectra integrates f, where f is axis 0: components, axis 1: k1, axis 2: k2, axis 3: k3
        # The integration in OnePointSpectra is over axis=2 (k3) then axis=1 (k2).
        # So, 'f' passed to OnePointSpectra.__call__ should have shape like (num_components, Nk1, Nk2, Nk3).
        # Or, OnePointSpectra should take one component at a time.
        # The current OnePointSpectra.__call__(self, f) takes 'f' and integrates.
        # If 'f' is one component (Nk1, Nk2, Nk3), it returns (Nk1).
        # The original PyTorch code: `kF = torch.stack([ k1_input * self.quad23(Phi) for Phi in self.Phi ])`
        # This implies `self.quad23` (our JAX OnePointSpectra.__call__) is called per component.
        # So RDT_spectral_tensor.__call__ should return the tuple of (Nk1,Nk2,Nk3) tensors.
        return phi_components

    def eddy_lifetime_initial_guess(self, k: jnp.ndarray) -> jnp.ndarray:
        """Compute initial guess for the eddy lifetime function."""
        return 0


#############################################################################################################
# Define objective calculation (OnePointSpectra)
#############################################################################################################


# Ok, the goal here is to go from the eddy lifetime function to the one point spectra.

# Phi(k, \tau_DRD) = D_(\tau_DRD)(k) \Phi_VK(k) D^*_(\tau_DRD)(k)

@dataclass
class k2_k3_parameters:
    """Parameters defining the logarithmic k2/k3 grids used for integration."""

    k2_min_p: int = -3
    k2_max_p: int = 3
    k2_points: int = 100

    k3_min_p: int = -3
    k3_max_p: int = 3
    k3_points: int = 100

    def __post_init__(self):
        assert self.k2_points > 0
        assert self.k3_points > 0
        assert self.k2_min_p < self.k2_max_p
        assert self.k3_min_p < self.k3_max_p


class OnePointSpectra(eqx.Module):
    """
    Compose this with a TauNet and a spectral tensor model.

    NOTE: This returns kF, not F
    """

    phi_drd: RDT_spectral_tensor
    k2_k3_params: k2_k3_parameters

    _k2_coords: jnp.ndarray = eqx.static_field()
    _k3_coords: jnp.ndarray = eqx.static_field()

    def __init__(self, phi_drd: RDT_spectral_tensor, k2_k3_params: k2_k3_parameters):

        object.__setattr__(self, "phi_drd", phi_drd)
        object.__setattr__(self, "k2_k3_params", k2_k3_params)

        k2_coords = jnp.logspace(
            k2_k3_params.k2_min_p,
            k2_k3_params.k2_max_p,
            k2_k3_params.k2_points,
        )
        k3_coords = jnp.logspace(
            k2_k3_params.k3_min_p,
            k2_k3_params.k3_max_p,
            k2_k3_params.k3_points,
        )

        object.__setattr__(self, "_k2_coords", k2_coords)
        object.__setattr__(self, "_k3_coords", k3_coords)

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """Integrate the 9 components of the DRD spectral tensor."""
        integral_over_k3 = jax.scipy.integrate.trapezoid(
            f,
            x = self._k3_coords,
            axis = 2
        )

        integral_over_k2_k3 = jax.scipy.integrate.trapezoid(
            integral_over_k3,
            x = self._k2_coords,
            axis = 1
        )

        return integral_over_k2_k3


#############################################################################################################
# Loss functions and training loop
#############################################################################################################

@jax.jit
def log_mse_term(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean-squared error on *log* scale (as in original PyTorch calibration).

    Both *y_pred* and *y_true* must be strictly positive / non-zero.  A small
    epsilon is applied internally to avoid `log(0)`.
    """
    eps = 1.0e-12
    return jnp.mean(jnp.square(jnp.log(jnp.abs(y_pred) + eps) - jnp.log(jnp.abs(y_true) + eps)))


def compute_kF_components(rdt_model: RDT_spectral_tensor, integrator: OnePointSpectra, k1_vec: jnp.ndarray) -> jnp.ndarray:
    """Compute *k₁ F* components (11,22,33,13) for a batch of *k₁* values."""
    phi = rdt_model(k1_vec)  # tuple of 6 tensors, shape [Nk1, Nk2, Nk3]

    kF_list = []
    for idx in (0, 1, 2, 3):  # Phi11, Phi22, Phi33, Phi13
        integ = integrator(phi[idx])  # (Nk1,)
        kF_list.append(k1_vec * integ)

    return jnp.stack(kF_list)  # (4, Nk1)


def loss_DRD(params, model_template: RDT_spectral_tensor, integrator: OnePointSpectra, k1_vec: jnp.ndarray, y_data: jnp.ndarray) -> jnp.ndarray:
    """Total loss for DRD calibration (only log-MSE term for now)."""
    # Re‐inject the parameter PyTree into the model
    rdt_model = eqx.combine(model_template, params)

    y_pred = compute_kF_components(rdt_model, integrator, k1_vec)

    return log_mse_term(y_pred, y_data)


#############################################################################################################
# Training loop (toy – Adam optimiser)
#############################################################################################################

@dataclass
class TrainingParameters:
    num_epochs: int = 2000
    lr: float = 1e-3

def training_loop(num_epochs: int = 2000, lr: float = 1e-3, key: Optional[jax.random.PRNGKey] = None):
    """Simple training loop using Optax Adam to calibrate the DRD model on Kaimal toy data."""
    if key is None:
        key = jax.random.PRNGKey(0)

    # ------------------------------------------------------------
    # Generate synthetic data (Kaimal)
    # ------------------------------------------------------------
    import drdmannturb.data_generator as jdg

    zref = 40.0
    ustar = 1.773
    k1_vec = jnp.logspace(-1, 2, 60) / zref  # (Nk1,)

    data_dict = jdg.generate_kaimal_data(k1_vec, zref, ustar)
    phi_true = data_dict["phi"]  # (Nk1, 3, 3)

    # Build target kF array with components (11,22,33,13)
    kF_target = jnp.stack(
        [
            k1_vec * phi_true[:, 0, 0],
            k1_vec * phi_true[:, 1, 1],
            k1_vec * phi_true[:, 2, 2],
            k1_vec * phi_true[:, 0, 2],
        ]
    )  # (4, Nk1)

    # ------------------------------------------------------------
    # Instantiate model objects
    # ------------------------------------------------------------
    tau_net = TauNet()
    k23_params = k2_k3_parameters()
    rdt_template = RDT_spectral_tensor(tau_net, k2_k3_params=k23_params)
    integrator = OnePointSpectra(rdt_template, k23_params)

    # Separate parameters and static parts for Equinox
    params, static = eqx.partition(rdt_template, eqx.is_array)

    # ------------------------------------------------------------
    # Optax optimiser
    # ------------------------------------------------------------
    optim = optax.adam(lr)
    opt_state = optim.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_DRD)(
            params, static, integrator, k1_vec, kF_target
        )
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(num_epochs):
        params, opt_state, loss_val = step(params, opt_state)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} – loss {loss_val:.6e}")

    # Merge parameters back into a full model instance
    calibrated_model = eqx.combine(static, params)
    print("Training complete – final loss", loss_val)
    return calibrated_model


#############################################################################################################
# Utility – plotting
#############################################################################################################

def plot_kf_results(
    model: RDT_spectral_tensor,
    integrator: OnePointSpectra,
    k1_vec: jnp.ndarray,
    kF_target: jnp.ndarray,
    save_path: str | None = None,
):
    """Plot calibrated vs. target $k_1 F_{ij}$ curves.

    Parameters
    ----------
    model : RDT_spectral_tensor
        Calibrated DRD spectral tensor model.
    integrator : OnePointSpectra
        Integrator instance (must use same k2/k3 grids as `model`).
    k1_vec : jnp.ndarray
        1-D array of $k_1$ frequencies.
    kF_target : jnp.ndarray
        Reference curves with shape (4, Nk1) corresponding to components
        (11, 22, 33, 13).
    save_path : str | None
        If provided, figure is saved to this path; otherwise displayed.
    """
    # Get model predictions (CPU numpy for Matplotlib)
    kF_pred = compute_kF_components(model, integrator, k1_vec)
    k1_np = jnp.asarray(k1_vec).astype(float)
    pred_np = jnp.asarray(kF_pred).astype(float)
    tgt_np = jnp.asarray(kF_target).astype(float)

    labels = [r"$F_{11}$", r"$F_{22}$", r"$F_{33}$", r"$F_{13}$"]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(4):
        ax.plot(k1_np, pred_np[i], color=colors[i], lw=2, label=labels[i] + " – model")
        ax.scatter(k1_np, tgt_np[i], color=colors[i], marker="o", facecolors="none", label=labels[i] + " – data")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_1$ [1/m]")
    ax.set_ylabel(r"$k_1 F_{ij}$ [-]")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":

    import drdmannturb.data_generator as jdg

    zref = 40.0
    ustar = 1.773
    k1_vec = jnp.logspace(-1, 2, 60) / zref
    data_dict = jdg.generate_kaimal_data(k1_vec, zref, ustar)
    kF_target = jnp.stack(
        [
            k1_vec * data_dict["phi"][:, 0, 0],
            k1_vec * data_dict["phi"][:, 1, 1],
            k1_vec * data_dict["phi"][:, 2, 2],
            k1_vec * data_dict["phi"][:, 0, 2],
        ]
    )

    # run training (it can reuse the same k1_vec, kF_target you just made)
    calib = training_loop()

    # need an integrator instance that matches the trained model
    k23_params = k2_k3_parameters()
    integrator = OnePointSpectra(calib, k23_params)

    plot_kf_results(calib, integrator, k1_vec, kF_target)
