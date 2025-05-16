"""Generate low-frequency fluctuation field, meant as an augmentation to other 3d fields.

The methods implemented here are based on:
[1] A.H. Syed, J. Mann "A Model for Low-Frequency, Anisotropic Wind Fluctuations and Coherences
    in the Marine Atmosphere", Boundary-Layer Meteorology 190:1, 2024 <https://doi.org/10.1007/s10546-023-00850-w>
[2] A.H. Syed, J. Mann "Simulating low-frequency wind fluctuations", Wind Energy Science, 9, 1381-1391, 2024
    <https://doi.org/10.5194/wes-9-1381-2024>
"""

from typing import Dict

LowFreqConfig = Dict[str, float]
