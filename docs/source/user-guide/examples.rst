Examples and Tutorials
=====================

This section contains working examples and tutorials for DRDMannTurb. All examples are automatically generated from the source code and can be run interactively.

.. toctree::
   :maxdepth: 2
   :caption: Gallery:

   ../auto_examples/index

.. raw:: html

    <hr>

Spectra Fitting Examples
------------------------

These examples demonstrate basic workflows and are perfect for getting started with DRDMannTurb.

.. toctree::
   :maxdepth: 1
   :caption: Basic Examples:

   ../auto_examples/01_mann_elt_fit
   ../auto_examples/02_taunet_elt_fit
   ../auto_examples/03_custom_noisy_data_fit

.. raw:: html

    <hr>

Field Generation Examples
-------------------------

These examples show how to generate synthetic turbulent fields using calibrated models.

.. toctree::
   :maxdepth: 1
   :caption: Field Generation:

   ../auto_examples/04_mann_box_generation_IEC
   ../auto_examples/05_drd_box
   ../auto_examples/06_mann_2d_plus_3d

.. raw:: html

    <hr>

Running Examples Locally
------------------------

To run these examples locally:

1. **Install DRDMannTurb**:
   .. code-block:: bash

      pip install drdmannturb

2. **Clone the repository**:
   .. code-block:: bash

      git clone https://github.com/METHODS-Group/DRDMannTurb.git
      cd DRDMannTurb

3. **Run an example**:
   .. code-block:: bash

      python examples/01_mann_elt_fit.py

4. **Interactive execution**:
   You can also run examples interactively in Jupyter:
   .. code-block:: python

      %run examples/01_mann_elt_fit.py

Tips for Running Examples
------------------------

**Environment Setup**
   Make sure you have all dependencies installed:
   .. code-block:: bash

      pip install torch numpy scipy matplotlib

**GPU Acceleration**
   For faster execution, use GPU if available:
   .. code-block:: python

      import torch
      if torch.cuda.is_available():
          device = "cuda"
      else:
          device = "cpu"

**Memory Management**
   For large examples, monitor memory usage:
   .. code-block:: python

      import psutil
      print(f"Memory usage: {psutil.virtual_memory().percent}%")

**Debugging**
   Enable verbose output for debugging:
   .. code-block:: python

      results = problem.calibrate(verbose=True)

Next Steps
----------

- Read the :doc:`spectra-fitting` guide for detailed explanations
- Explore the :doc:`field-generation` guide for advanced features
- Review the :doc:`../api/spectra-fitting` for complete API reference
- Check out the `examples/ <https://github.com/METHODS-Group/DRDMannTurb/tree/main/examples>`_ directory for source code
