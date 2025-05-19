"""Generate low-frequency fluctuation field, meant as an augmentation to other 3d fields.

The methods implemented here are based on:
[1] A.H. Syed, J. Mann "A Model for Low-Frequency, Anisotropic Wind Fluctuations and Coherences
    in the Marine Atmosphere", Boundary-Layer Meteorology 190:1, 2024 <https://doi.org/10.1007/s10546-023-00850-w>
[2] A.H. Syed, J. Mann "Simulating low-frequency wind fluctuations", Wind Energy Science, 9, 1381-1391, 2024
    <https://doi.org/10.5194/wes-9-1381-2024>
"""

from dataclasses import dataclass


@dataclass
class LowFreqGenerator_Params:

    sigma2: float # Target variance of the low-frequency fluctuations
    L_2d: float # 2D integral scale
    psi: float # Anisotropy parameter
    z_i: float # Attenuation length, assumed to be height of atmospheric boundary layer

    c: float # Scaling constant, which is calculated from the parameters above

    Lx: float # Length of the domain in x-direction
    Ly: float # Length of the domain in y-direction

    def __post_init__(self):
        # REPLACE
        self.c = self.sigma2 / (self.L_2d ** 2 * self.z_i ** 2)





def generate_low_freq_fluctuation_field(
    params: LowFreqGenerator_Params
):
    
    pass


