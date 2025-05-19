from dataclasses import dataclass

import jax.numpy as jnp
import jaxtyping as jt


@dataclass
class WindGenerator_State:


    # length 3 array of positive integers, denoting 2^N + 1 grid points in respective direction
    grid_levels: jt.Int












