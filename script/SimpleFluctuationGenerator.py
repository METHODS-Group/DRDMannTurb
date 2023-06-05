
from math import pi
from matplotlib.pyplot import axis
import numpy as np
import scipy.fft as fft
from pyevtk.hl import imageToVTK


"""
=============================================
Configuration
=============================================
"""

ndim        = 3
grid_levels = np.array([7,7,5]) #np.array([8]*ndim)
grid_shape  = 2**grid_levels
domain_size = np.array([1, 1, 0.25 ]) #[1]*ndim

friction_velocity = 2.683479938442173
reference_height  = 180.0
roughness_height  = 0.75
prefactor         = 3.2 * friction_velocity**2 * reference_height**(-2/3)
Lengthscale       = 0.59 * reference_height



"""
=============================================
Spectral tensor
=============================================
"""
def VonKarmanSpectralTenor(prefactor=1, beta=None):

    Nd, d          = grid_shape.copy(), ndim
    frequences     = [fft.fftfreq(Nd[j], domain_size[j]/Nd[j]) for j in range(d)]
    Nd[-1]         = int(Nd[-1] // 2)+1 ### the last dimension for rfft is twice shorter
    frequences[-1] = frequences[-1][:Nd[-1]]

    k  = np.array(list(np.meshgrid(*frequences, indexing='ij')))
    kk = np.sum(k**2, axis=0)

    ### curl
    SqrtSpectralTens = np.tile(np.zeros(Nd, dtype=np.complex128),(3,3,1,1,1)) 
    SqrtSpectralTens[0,1,...] = -k[2,...]
    SqrtSpectralTens[0,2,...] =  k[1,...]
    SqrtSpectralTens[1,0,...] =  k[2,...]
    SqrtSpectralTens[1,2,...] = -k[0,...]
    SqrtSpectralTens[2,0,...] = -k[1,...]
    SqrtSpectralTens[2,1,...] =  k[0,...]
    SqrtSpectralTens *= 1j

    ### energy spectrum
    L                  = Lengthscale
    const              = prefactor * (L**(17/3)) / (4*pi)
    SqrtEnergySpectrum = np.sqrt( const / (1 + (L**2) * kk)**(17/6) )
    SqrtSpectralTens  *= SqrtEnergySpectrum

    ### dissipation (due to diffusion)
    if beta: SqrtSpectralTens *= np.exp(-beta*np.sqrt(kk))

    return SqrtSpectralTens


"""
=============================================
Generate fluctuation field
=============================================
"""

noise           = np.random.normal(loc=0, scale=1, size=[ndim]+list(grid_shape))
noise_hat       = fft.rfftn(noise, norm="ortho")
kernel_hat      = VonKarmanSpectralTenor(prefactor=prefactor)
fluctuation_hat = (kernel_hat * noise_hat).sum(axis=1) ### matrix-vector product
fluctuation     = fft.irfftn(fluctuation_hat, norm="ortho")

"""
=============================================
Export vtk
=============================================
"""
FileName = './workfolder/fluctuation'
spacing  = list(domain_size/grid_shape) #[1/grid_shape.max()]*ndim
cellData = {'fluctuation field': tuple(fluctuation)}
imageToVTK(FileName, cellData = cellData, spacing=spacing)