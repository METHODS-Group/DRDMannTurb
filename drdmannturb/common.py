"""
This module contains implementations of common functions, specifically the Mann eddy lifetime function
and the von Karman energy spectrum.
"""

__all__ = ["VKEnergySpectrum", "MannEddyLifetime", "Mann_linear_exponential_approx"]

import io
import pickle
from dataclasses import astuple
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import hyp2f1
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@torch.jit.script
def VKEnergySpectrum(kL: torch.Tensor) -> torch.Tensor:
    r"""
    Von Karman energy spectrum without scaling:

    .. math::
        \widetilde{E}(\boldsymbol{k}) = \left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3}.

    Parameters
    ----------
    kL : torch.Tensor
        Scaled wave number domain.

    Returns
    -------
    torch.Tensor
        Result of the evaluation
    """
    return kL**4 / (1.0 + kL**2) ** (17.0 / 6.0)


def MannEddyLifetime(kL: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    r"""
    Implementation of the full Mann eddy lifetime function, of the form

    .. math::
        \tau^{\mathrm{IEC}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1/3, 17/6; 4/3 ;-(kL)^{-2}\right)}}

    This function can execute with input data that are either in Torch or numpy. However,

    .. warning::
        This function depends on SciPy for evaluating the hypergeometric function, meaning a GPU tensor will be returned
        to the CPU for a single evaluation and then converted back to a GPU tensor. This incurs a substantial loss of
        performance.

    Parameters
    ----------
    kL : Union[torch.Tensor, np.ndarray]
        Scaled wave number

    Returns
    -------
    torch.Tensor
        Evaluated Mann eddy lifetime function.
    """

    x = kL.cpu().detach().numpy() if torch.is_tensor(kL) else kL
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(x ** (-2))))
    y = torch.tensor(y, dtype=torch.float64) if torch.is_tensor(kL) else y

    return y


def Mann_linear_exponential_approx(
    kL: torch.Tensor, coefficient: torch.Tensor, intercept: torch.Tensor
) -> torch.Tensor:
    r"""A surrogate for the term involving the hypergeometric function

    .. math::
        \frac{x^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-x^{-2}\right)}}

    via an exponential function, which is a reasonable approximation since the resulting :math:`\tau` is nearly linear
    on a log-log plot. The resulting approximation is the function

    .. math::
        \exp \left( \alpha kL + \beta \right)

    where :math:`\alpha, \beta` are obtained from a linear regression on the hypergeometric function on the domain of
    interest. In particular, using this function requires that a linear regression has already been performed on the
    basis of the above function depending on the hypergeometric function, which is an operation performed once on the
    CPU. The rest of this subroutine is on the GPU and unlike the full hypergeometric approximation, will not incur
    any slow-down of the rest of the spectra fitting.

    Parameters
    ----------
    kL : torch.Tensor
        _description_
    coefficient : torch.Tensor
        The linear coefficient :math:`\alpha` obtained from the linear regression.
    intercept : torch.Tensor
        The intercept term :math:`\beta` from the linear regression.

    Returns
    -------
    torch.Tensor
        Exponential approximation to the Mann eddy lifetime function output.
    """

    return torch.exp(coefficient * torch.log(kL) + intercept)


class CPU_Unpickler(pickle.Unpickler):
    """Utility for loading tensors onto CPU; credit: https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219"""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def plot_loss_logs(log_file: Union[str, Path]):
    """Returns a full plot of all loss terms for a specific training log generated by TensorBoard. This is an auxiliary
    method and should only be used for quick visualization of the training process, the suggested method for visualizing
    this information is through the TensorBoard API.

    Parameters
    ----------
    log_file : Union[str, Path]
       Full path to training log to be visualized.
    """

    event_acc = EventAccumulator(log_file)
    event_acc.Reload()

    training_scalars = event_acc.Tags()["scalars"]

    vals_tot = {}
    for t_scalar in training_scalars:
        _, _, vals_curr = zip(*[astuple(event) for event in event_acc.Scalars(t_scalar)])

        vals_tot[t_scalar] = vals_curr[1:]

    with plt.style.context("bmh"):
        plt.rcParams.update({"font.size": 8})
        fig, ax = plt.subplots(1, vals_tot.__len__(), figsize=(12, 4), sharex=True)
        for idx, (scalar_tag, vals) in enumerate(reversed(vals_tot.items())):
            ax[idx].plot(vals[1:])
            ax[idx].set_yscale("log")
            ax[idx].set_title(scalar_tag)

        fig.text(0.5, 0.01, "Wolfe Iterations", ha="center")
        fig.tight_layout()

    plt.show()

    return
