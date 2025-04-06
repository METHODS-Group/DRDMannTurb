
-----------------------
Is there a better way for me to rearrnage the analytical_Fij, or maybe I should avoid using integrate.quad and we can roll our own quadrature? I am trying to keep this relatively fast, and repeating a bunch of calculations is one way to avoid this.

I am trying to compute for F11 and F22 = \int_{-\infty}^{\infty} \Phi_{11}(k) dk2 numerically, since there does not exist a closed form of the integral.

Phi_ij(k) = E( kappa(k, psi) ) * P_ij (k) / pi * k

Assume the definitions of Ekappa and kappa are correct. Pij is the "projection tensor" delta_[ij} - k_i k_j / k^2





------------------
