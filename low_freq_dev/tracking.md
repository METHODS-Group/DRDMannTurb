
# Debugging the spectrum match

BTW the points are sparse for k1L < 1 because the wavenumbers are
uniformly spaced. Reverse idea for k1L > 1.

###### Currently correct

- Past 2^7 in either direction, the method seems to converge. Below this number,
  it's basically fine still.
- Method is similarly independent of physical lengths L1 and L2.
   - The number here though is 5xL. There is maybe some physical reason wrt parameters
     that I have butchered lol.
- The peak of estimated spectrum is at k1L = 1 for both F11 and F22.


###### Current maybe's

- The slopes are 3/4 OK


###### Current issues

- F11 k1L > 1 is too steep
- Scale of spectrum is way too small
- Fluctuation fields seem to be losing a factor of 10
- Simulated spectrum is only over k1L >= 1, where we want at least 10^-3 to 10^3

- The spectrum should be independent of domain size...


## 1. How do we get the estimated spectrum to "extend left"?

Play with length scale multiple `L1_factor` and `L2_factor` parameters.
