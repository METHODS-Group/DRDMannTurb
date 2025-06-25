.. py:currentmodule:: drdmannturb

Data Types and Eddy Lifetime Functions
======================================

``DRDMannTurb`` uses several enumerations to set information regarding the type of spectra data being used, the eddy lifetime model to use, and the power spectra law (there is currently only one implemented: the RDT power spectra). The following is an exhaustive list of their labels and values, with individual discussion further below.

Data Types
----------

The type of spectral tensor is determined by the :py:enum:`DataType` enum, which determines one of the following models:

#. ``DataType.KAIMAL``, which is the Kaimal spectra.

#. ``DataType.VK``, which is the von Karman spectra model.

#. ``DataType.CUSTOM``, usually used for data that is processed from real-world data. The spectra values are to be provided as the ``spectra_values`` field, or else to be loaded from a provided ``spectra_file``. The result is that the provided data are matched on the wavevector domain.

Eddy Lifetime Function
----------------------

The type of eddy lifetime function :math:`\tau` to use, its form, and whether or not to learn this function is determined by :py:enum:`EddyLifetimeType`. The following models are provided, with more discussions of each in the next section:

#. ``EddyLifetimeType.CONST`` which simply returns :math:`\tau = 1`.

#. ``EddyLifetimeType.TWOTHIRD`` which returns :math:`\tau = (kL)^{-2/3}`.

#. ``EddyLifetimeType.MANN``, which is the Mann function determined by

    .. math::

        \tau^{\mathrm{Mann}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    .. note::
        This is a CPU-bound function since the hypergeometric function has no readily use-able CUDA implementation, which considerably impacts performance. Consider using an approximation to this function.

#. ``EddyLifetimeType.MANN_APPROX``, which performs a linear regression on a single evaluation of the Mann eddy lifetime function in log-log space. This results in a GPU function of the form :math:`\tau \approx \exp (\alpha \boldsymbol{k} + \beta)`.

#. ``EddyLifetimeType.TAUNET``, which performs a simple neural-network based fit of the eddy lifetime function. This has the component of learning

    .. math::
        \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}{\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}).

    The architecture here is identical to the one used in the original DRD paper: 2 layers of 10 neurons each, connected by a RELU function. Setting ``learn_nu=True`` means the neural network will also attempt to learn the :math:`\nu` value.

#. ``EddyLifetimeType.CUSTOMMLP`` which is an expanded API of the ``TAUNET``: the number of layers, widths of each layer, and activation functions between each layer can be specified. The ``learn_nu`` variable serves the same function as for ``TAUNET``.

.. autoenum:: drdmannturb.EddyLifetimeType
    :members:

Mann Eddy Lifetime Function and VK Energy Spectrum
--------------------------------------------------

The associated eddy lifetime functions are as follows:

.. py:currentmodule:: drdmannturb.common

.. autofunction:: MannEddyLifetime

.. autofunction:: Mann_linear_exponential_approx

.. autofunction:: VKEnergySpectrum
