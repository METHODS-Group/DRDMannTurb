"""
This modules implements built-in plotting utilities for ease of use
"""

import matplotlib.pyplot as plt
from typing import Any

class PlotSpectrum:
    def __init__(self, **kwargs) -> None:
        plt_dynamic = kwargs.get("plt_dynamic", False)
        plt_tau = kwargs.get("plt_tau", True)

        k1 = self.k1_data_pts
        k = torch.stack([k1, 0 * k1, 0 * k1], dim=-1)
        if plt_tau:
            k_norm = torch.norm(k, dim=-1)

        if plt_dynamic:
            ion()
        else:
            ioff()

        self.kF_model_vals = kwargs.get("model_vals", None)
        if self.kF_model_vals is None:
            self.kF_model_vals = self.OPS(k1).detach().numpy()

        if not hasattr(self, "fig"):
            nrows = 1
            ncols = 2 if plt_tau else 1
            self.fig, self.ax = subplots(
                nrows=nrows,
                ncols=ncols,
                num="Calibration",
                clear=True,
                figsize=[20, 10],
            )
            if not plt_tau:
                self.ax = [self.ax]

            ### Subplot 1: One-point spectra
            self.ax[0].set_title("One-point spectra")
            self.lines_SP_model = [None] * (self.vdim + 1)
            self.lines_SP_data = [None] * (self.vdim + 1)

            for i in range(self.vdim):
                (self.lines_SP_model[i],) = self.ax[0].plot(
                    k1,
                    self.kF_model_vals[i],
                    "o-",
                    label=r"$F{0:d}$ model".format(i + 1),
                )
            for i in range(self.vdim):
                (self.lines_SP_data[i],) = self.ax[0].plot(
                    k1, self.kF_data_vals[i], "--", label=r"$F{0:d}$ data".format(i + 1)
                )
            # self.lines_SP_model[self.vdim], = self.ax[0].plot(k1, -self.kF_model_vals[self.vdim].detach().numpy(), 'o-', label=r'$-F_{13}$ model')
            # self.lines_SP_data[self.vdim],  = self.ax[0].plot(k1, -self.kF_data_vals[self.vdim].detach().numpy(), '--', label=r'$-F_{13}$ data')
            self.ax[0].legend()
            self.ax[0].set_xscale("log")
            self.ax[0].set_yscale("log")
            self.ax[0].set_xlabel(r"$k_1$")
            self.ax[0].set_ylabel(r"$k_1 F_i$")
            self.ax[0].grid(which="both")
            self.ax[0].set_aspect(3 / 4)
            # self.ax[0].yaxis.set_minor_formatter(FormatStrFormatter())
            # self.ax[0].yaxis.set_major_formatter(FormatStrFormatter())
            # self.ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # self.ax[0].yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
            # self.ax[0].set_yticks(ticks=[0.01,0.02,0.05,0.1,0.2,0.5], minor=True)

            if plt_tau:
                ### Subplot 2: Eddy Lifetime
                self.ax[1].set_title("Eddy liftime")
                self.tau_model = self.OPS.EddyLifetime(k).detach().numpy()
                self.tau_ref = 3.9 * MannEddyLifetime(0.59 * k_norm).detach().numpy()
                (self.lines_LT_model,) = self.ax[1].plot(
                    k_norm, self.tau_model, "-", label=r"$\tau_{model}$"
                )
                (self.lines_LT_ref,) = self.ax[1].plot(
                    k_norm, self.tau_ref, "--", label=r"$\tau_{ref}=$Mann"
                )
                self.ax[1].legend()
                # self.ax[1].set_aspect(3/4)
                self.ax[1].set_xscale("log")
                # self.ax[1].set_yscale('log')
                self.ax[1].set_xlabel(r"$k$")
                self.ax[1].set_ylabel(r"$\tau$")
                self.ax[1].grid(which="both")

        (hl,) = plt.subplot([], [])

    def update(self, new_data):
        for i in range(self.vdim):
            self.lines_SP_model[i].set_ydata(self.kF_model_vals[i])
        hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
        hl.set_ydata(numpy.append(hl.get_ydata(), new_data))

        if self.fg_tau:
            self.tau_model = self.OPS.EddyLifetime(k).detach().numpy()
            self.lines_LT_model.set_ydata(self.tau_model)

        plt.draw()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.update()


def plot(self, **kwargs):
    plt_dynamic = kwargs.get("plt_dynamic", False)
    plt_tau = kwargs.get("plt_tau", True)

    k1 = self.k1_data_pts
    k = torch.stack([k1, 0 * k1, 0 * k1], dim=-1)
    if plt_tau:
        k_norm = torch.norm(k, dim=-1)

    if plt_dynamic:
        ion()
    else:
        ioff()

    self.kF_model_vals = kwargs.get("model_vals", None)
    if self.kF_model_vals is None:
        self.kF_model_vals = self.OPS(k1).detach().numpy()

    if not hasattr(self, "fig"):
        nrows = 1
        ncols = 2 if plt_tau else 1
        self.fig, self.ax = subplots(
            nrows=nrows, ncols=ncols, num="Calibration", clear=True, figsize=[20, 10]
        )
        if not plt_tau:
            self.ax = [self.ax]

        ### Subplot 1: One-point spectra
        self.ax[0].set_title("One-point spectra")
        self.lines_SP_model = [None] * (self.vdim + 1)
        self.lines_SP_data = [None] * (self.vdim + 1)

        for i in range(self.vdim):
            (self.lines_SP_model[i],) = self.ax[0].plot(
                k1, self.kF_model_vals[i], "o-", label=r"$F{0:d}$ model".format(i + 1)
            )
        for i in range(self.vdim):
            (self.lines_SP_data[i],) = self.ax[0].plot(
                k1, self.kF_data_vals[i], "--", label=r"$F{0:d}$ data".format(i + 1)
            )
        # self.lines_SP_model[self.vdim], = self.ax[0].plot(k1, -self.kF_model_vals[self.vdim].detach().numpy(), 'o-', label=r'$-F_{13}$ model')
        # self.lines_SP_data[self.vdim],  = self.ax[0].plot(k1, -self.kF_data_vals[self.vdim].detach().numpy(), '--', label=r'$-F_{13}$ data')
        self.ax[0].legend()
        self.ax[0].set_xscale("log")
        self.ax[0].set_yscale("log")
        self.ax[0].set_xlabel(r"$k_1$")
        self.ax[0].set_ylabel(r"$k_1 F_i$")
        self.ax[0].grid(which="both")
        self.ax[0].set_aspect(3 / 4)
        # self.ax[0].yaxis.set_minor_formatter(FormatStrFormatter())
        # self.ax[0].yaxis.set_major_formatter(FormatStrFormatter())
        # self.ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # self.ax[0].yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        # self.ax[0].set_yticks(ticks=[0.01,0.02,0.05,0.1,0.2,0.5], minor=True)

        if plt_tau:
            ### Subplot 2: Eddy Lifetime
            self.ax[1].set_title("Eddy liftime")
            self.tau_model = self.OPS.EddyLifetime(k).detach().numpy()
            self.tau_ref = 3.9 * MannEddyLifetime(0.59 * k_norm).detach().numpy()
            (self.lines_LT_model,) = self.ax[1].plot(
                k_norm, self.tau_model, "-", label=r"$\tau_{model}$"
            )
            (self.lines_LT_ref,) = self.ax[1].plot(
                k_norm, self.tau_ref, "--", label=r"$\tau_{ref}=$Mann"
            )
            self.ax[1].legend()
            # self.ax[1].set_aspect(3/4)
            self.ax[1].set_xscale("log")
            # self.ax[1].set_yscale('log')
            self.ax[1].set_xlabel(r"$k$")
            self.ax[1].set_ylabel(r"$\tau$")
            self.ax[1].grid(which="both")

    for i in range(self.vdim):
        self.lines_SP_model[i].set_ydata(self.kF_model_vals[i])
    # self.lines_SP_model[self.vdim].set_ydata(-self.kF_model_vals[self.vdim].detach().numpy())
    self.ax[0].set_aspect(3 / 4)

    if plt_tau:
        self.tau_model = self.OPS.EddyLifetime(k).detach().numpy()
        self.lines_LT_model.set_ydata(self.tau_model)

    if plt_dynamic:
        self.fig.gca().autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    else:
        self.fig.show()
        plt.show()
