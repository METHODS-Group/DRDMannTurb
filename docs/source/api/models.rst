Models API
==========

Neural network models and parameter classes for DRDMannTurb.

Neural Network Modules
----------------------

Neural network architectures for learning eddy lifetime functions.

.. automodule:: drdmannturb.nn_modules
   :members:
   :undoc-members:

   **Key Classes:**

   - ``TauNet``: Neural network for eddy lifetime function
   - ``Rational``: Rational kernel for enforcing analytic behavior

   **TauNet Architecture:**

   The TauNet combines a multi-layer perceptron with a rational kernel:

   .. math::
      \tau(\boldsymbol{k}) = \frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
      {(1+|\boldsymbol{a}|^2)^{\nu/2}}

   where :math:`\boldsymbol{a}(\boldsymbol{k}) = |\boldsymbol{k}| + \mathrm{NN}(|\boldsymbol{k}|)`

   **Example Usage:**

   .. code-block:: python

      from drdmannturb.nn_modules import TauNet

      # Create TauNet
      taunet = TauNet(
          n_layers=2,
          hidden_layer_sizes=[10, 10],
          learn_nu=True,
          nu_init=-1.0/3.0
      )

      # Use in spectral tensor model
      from drdmannturb.spectra_fitting.spectral_tensor_models import TauNet_ELT
      elt_model = TauNet_ELT(taunet)

Parameters
----------

Parameter classes for configuring the loss function and numerical integration. The physical
scales ``L``, ``Gamma``, and ``sigma`` are not separate parameter objects: they are learnable
parameters of the ``SpectralTensorModel`` itself, provided as ``L_init``/``gamma_init``/``sigma_init``
at model construction.

.. automodule:: drdmannturb.parameters
   :members:
   :undoc-members:
   :no-index:

   **Key Classes:**

   - ``LossParameters``: Loss function term coefficients
   - ``IntegrationParameters``: Quadrature grids for spectra and coherence integration

   **Example Usage:**

   .. code-block:: python

      from drdmannturb import IntegrationParameters, LossParameters

      loss_params = LossParameters(
          alpha_pen2=1.0,  # second-order penalty coefficient
          beta_reg=1e-5,   # network regularization coefficient
      )
      integration_params = IntegrationParameters()  # sensible defaults
