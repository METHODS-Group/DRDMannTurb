.. py:currentmodule:: drdmannturb

Data Types and Eddy Lifetime Functions
======================================

``DRDMannTurb`` uses several enumerations to set information regarding the type of spectra data being used, the eddy lifetime model to use, and the power spectra law (there is currently only one implemented: the power spectra). The following is an exhaustive list of their labels and values, with individual discussion further below. 

.. autoenum:: drdmannturb.DataType 
    :members: 

.. autoenum:: drdmannturb.PowerSpectraType

The power spectra law definition is given :doc:`here <./internal_rdt_ps>`.

.. autoenum:: drdmannturb.EddyLifetimeType
    :members: 

The associated eddy lifetime functions are as follows: 

.. py:currentmodule:: drdmannturb.common

.. autofunction:: MannEddyLifetime 

.. autofunction:: Mann_linear_exponential_approx

.. autofunction:: VKEnergySpectrum