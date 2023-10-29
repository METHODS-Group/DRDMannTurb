.. py:currentmodule:: drdmannturb

Internal API Reference
======================

Here is a documentation of methods internal to the package, which are subject to considerable change between releases. No promises of backwards compatibility are made with these methods.

Internal MLP and ResNet Interfaces
----------------------------------

.. autoclass:: drdmannturb.SimpleNN
    :members:

.. autoclass:: drdmannturb.Rational
    :members:

.. autoclass:: drdmannturb.CustomMLP
    :members:

.. autoclass:: drdmannturb.ResNetBlock
    :members:


Covariance Kernels 
------------------

.. autoclass:: drdmannturb.Covariance 
    :members: 

.. autoclass:: drdmannturb.VonKarmanCovariance
    :members

.. autoclass:: drdmannturb.MannCovariance
    :members

.. autoclass:: drdmannturb.NNCovariance
    :members


Covariance Sampling Methods
---------------------------

.. autoclass:: drdmannturb.Sampling_method_base
    :members

.. autoclass:: drdmannturb.Sampling_method_freq
    :members

.. autoclass:: drdmannturb.Sampling_FFTW
    :members

.. autoclass:: drdmannturb.Sampling_VF_FFTW
    :members

.. autoclass:: drdmannturb.Sampling_FFT
    :members

.. autoclass:: drdmannturb.Sampling_DST
    :members

.. autoclass:: drdmannturb.Sampling_DCT
    :members




Loss Functions for Calibration Problems
---------------------------------------

