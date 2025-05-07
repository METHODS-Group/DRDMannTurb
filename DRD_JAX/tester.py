import jax_data_generator as dg
import matplotlib.pyplot as plt

import numpy as np
import jax.numpy as jnp


def test_kaimal_spectra(k1_arr, z_ref):
    k1_arr = np.logspace(-3, 4, 500)
    z_ref = 50

    k1z = k1_arr * z_ref

    kaimal_data = dg.generate_kaimal_data(k1_arr, 50)
    print(type(kaimal_data))

    k1 = kaimal_data["k1"]
    phi = kaimal_data["phi"]
    coherence = kaimal_data["coherence"]

    assert len(k1) == phi.shape[0] == coherence.shape[0]


#####
# Going to plot the phi 00, 11, 22, and 02 components
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Kaimal Spectra Components")

# Plot phi_00 (uu)
axes[0, 0].loglog(k1z, phi[:, 0, 0])
axes[0, 0].set_title("phi_00 (uu)")
axes[0, 0].set_xlabel("k1*z")
axes[0, 0].set_ylabel("phi_00")

# Plot phi_11 (vv)
axes[0, 1].loglog(k1z, phi[:, 1, 1])
axes[0, 1].set_title("phi_11 (vv)")
axes[0, 1].set_xlabel("k1*z")
axes[0, 1].set_ylabel("phi_11")

# Plot phi_22 (ww)
axes[1, 0].loglog(k1z, phi[:, 2, 2])
axes[1, 0].set_title("phi_22 (ww)")
axes[1, 0].set_xlabel("k1*z")
axes[1, 0].set_ylabel("phi_22")

# Plot phi_02 (uw)
axes[1, 1].loglog(k1z, phi[:, 0, 2])
axes[1, 1].set_title("phi_02 (uw)")
axes[1, 1].set_xlabel("k1*z")
axes[1, 1].set_ylabel("phi_02")

plt.tight_layout()
plt.show()



if __name__ == "__main__":

    k1_arr = np.logspace(-3, 4, 500)
    z_ref = 50

    test_kaimal_spectra()
