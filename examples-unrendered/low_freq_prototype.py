from typing import Optional

import numpy as np
import scipy.integrate as integrate


class LowFreq2DFieldGenerator:
    def __init__(
        self,
        grid_dimensions: np.ndarray,
        grid_levels: np.ndarray,
        L_2D: float = 15_000.0,
        sigma2: float = 0.6,
        z_i: float = 500.0,
        psi_degs: float = 43.0,
        c: Optional[float] = None,
    ):
        # Field parameters
        self.L_2D = L_2D
        self.sigma2 = sigma2
        self.z_i = z_i
        self.psi_degs = psi_degs

        # Grid parameters
        self.L1, self.L2 = grid_dimensions[:2]  # Default was 60k x 15k
        self.Nx, self.Ny = 2 ** grid_levels[:2]  # Default was 1024 x 256

        self.psi_rad = np.deg2rad(self.psi_degs)

        if c is None:
            self.c = self._solve_for_c()
        else:
            self.c = c

    def _compute_kappa(self, kx, ky):
        """
        Compute the kappa value for a given kx, ky.
        """
        cos2 = np.cos(self.psi_rad) ** 2
        sin2 = np.sin(self.psi_rad) ** 2

        return np.sqrt(2.0 * ((kx**2) * cos2 + (ky**2) * sin2))

    def _compute_E(self, kappa: float) -> float:
        """
        Compute the energy spectrum E(kappa) for a given kappa.
        """
        if kappa < 1e-12:
            return 0.0
        denom = (1.0 / (self.L_2D**2) + kappa**2) ** (7.0 / 3.0)
        atten = 1.0 / (1.0 + (kappa * self.z_i) ** 2)
        return self.c * (kappa**3) / denom * atten

    def _solve_for_c(self):
        """
        Solve for scaling constant c so that integral of E(k) from k=0..inf = sigma2.
        """

        def integrand(k: float) -> float:
            return (k**3 / ((1.0 / (self.L_2D**2) + k**2) ** (7.0 / 3.0))) * (1.0 / (1.0 + (k * self.z_i) ** 2))

        val, _ = integrate.quad(integrand, 0, np.inf)
        return self.sigma2 / val

    def generate(
        self,
        pad: bool = True,
    ):
        """
        Generate a 2D low-frequency field.
        """
        L1, L2 = self.L1, self.L2
        Nx, Ny = self.Nx, self.Ny

        dx = L1 / Nx
        dy = L2 / Ny

        kx_arr = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky_arr = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
        kx_arr = np.fft.fftshift(kx_arr)  # sort from negative to positive
        ky_arr = np.fft.fftshift(ky_arr)

        Amp2 = np.zeros((Nx, Ny), dtype=np.float64)

        factor_16 = (2.0 * np.pi**2) / L1

        for ix in range(Nx):
            for iy in range(Ny):
                kx = kx_arr[ix]
                ky = ky_arr[iy]

                kappa = self._compute_kappa(kx, ky)
                E_val = self._compute_E(kappa)

                if kappa < 1e-12:
                    phi_11 = 0.0
                else:
                    phi_11 = E_val / (np.pi * kappa)

                amp2_kx = factor_16 * phi_11
                Amp2[ix, iy] = amp2_kx

        Uhat = np.zeros((Nx, Ny), dtype=np.complex128)
        for ix in range(Nx):
            for iy in range(Ny):
                amp = np.sqrt(Amp2[ix, iy])
                phase = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2.0)
                Uhat[ix, iy] = amp * phase

        Uhat_unshift = np.fft.ifftshift(Uhat, axes=(0, 1))
        u_field_complex = np.fft.ifft2(Uhat_unshift, s=(Nx, Ny))
        u_field = np.real(u_field_complex)

        var_now = np.var(u_field)
        if var_now > 1e-12:
            u_field *= np.sqrt(self.sigma2 / var_now)

        if pad:
            u_field = np.pad(u_field, ((0, 1), (0, 1)), mode="wrap")

        return np.linspace(0, L1 / 1000, Nx), np.linspace(0, L2 / 1000, Ny), u_field


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Domain: 60 km x 15 km
    L1 = 60_000.0
    L2 = 15_000.0
    # Nx = 1024 # = 2^10
    # Ny = 256 # = 2^8

    # Figure 3 parameters
    L2D = 15000.0  # [m]
    sigma2 = 0.6  # [m^2/s^2]
    z_i = 500.0  # [m]
    psi_degs = 43.0  # anisotropy angle

    generator = LowFreq2DFieldGenerator(
        grid_dimensions=np.array([L1, L2]),
        grid_levels=np.array([10, 8]),
        L_2D=L2D,
        sigma2=sigma2,
        z_i=z_i,
        psi_degs=psi_degs,
    )

    # Generate large-scale u-component
    _, _, u_field = generator.generate(pad=False)

    # Generate large-scale v-component similarly
    # (Here, we assume same sigma^2 and same approach.)
    x, y, v_field = generator.generate(pad=False)

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    im1 = axs[0].pcolormesh(X, Y, u_field, shading="auto", cmap="RdBu_r")
    cb1 = plt.colorbar(im1, ax=axs[0], label="m/s")
    axs[0].set_title("(a) u")
    axs[0].set_xlabel("x [km]")
    axs[0].set_ylabel("y [km]")

    im2 = axs[1].pcolormesh(X, Y, v_field, shading="auto", cmap="RdBu_r")
    cb2 = plt.colorbar(im2, ax=axs[1], label="m/s")
    axs[1].set_title("(b) v")
    axs[1].set_xlabel("x [km]")
    axs[1].set_ylabel("y [km]")

    plt.tight_layout()
    plt.show()
