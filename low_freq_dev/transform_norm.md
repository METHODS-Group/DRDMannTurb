# Transform norm notes

So, this is between fluctuation_field_generator, gaussian_random_fields, and sampling_methods.

In sampling_methods, we have
```
self.TransformNorm = np.sqrt(L.prod())
```
where `L` is `self.L` Sampling_VF_FFTW, which subclasses Sampling_method_freq, which subclasses Sampling_method_base, which requires a RandomField object for the constructor, which is where
`self.L = RandomField.L`, `self.Nd = RandomField.ext_grid_shape`, `self.ndim = RandomField.ndim` are set.

So, `sqrt(prod(RandomField.L))` is what we are interested in.

## RandomField module

We use VectorGaussianRandomField, which is a subclass of GaussianRandomField. VGRF doesn't do much extra.
```
self.ndim = ndim

...

# Grid level, shape are not scalar so we follow the second branch

h = grid_dimensions / (2**grid_level + 1)

self.grid_shape = np.array(grid_shape[:ndim])

self.L = h * self.grid_shape
```
So, `grid_dimensions`, `grid_shape`, and `grid_level`

## Fluctuation field generator module
```
# Initialize random field generator
self.RF = VectorGaussianRandomField(
    ndim=3,
    grid_level=grid_levels,
    grid_dimensions=grid_dimensions,
    sampling_method="vf_fftw",
    grid_shape=self.noise_shape[:-1],
    Covariance=self.Covariance,
)
```
So we need `grid_levels`, `grid_dimensions`, and `noise_shape[:-1]`

### `grid_levels`

This is the input `grid_levels`

### `grid_dimensions`

This is the input `grid_dimensions`

### `noise_shape[:-1]`

`self.noise_shape` is
```
self.noise_shape = [
    Nx + buffer_extension,
    Ny + margin_extension[0],
    Nz + margin_extension[1],
    3
]
```
So, `noise_shape[:-1]` is `[Nx + buffer_extension, Ny + margin_extension[0], Nz + margin_extension[1]]`.

`Nx, Ny, Nz` are exactly `2**grid_levels + 1`.

We set `blend_num` to `0`, so `buffer_extension = 2 * self.n_buffer` where
`self.n_buffer` is the somewhat arbitrary buffer size `ceil( (3 * Gamma * L) / dx)`

Now, `n_margin_y, n_margin_z` are just `ceil(L / dy), ceil(L / dz)`

