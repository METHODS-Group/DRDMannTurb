Quickstart Guide
===============

This guide provides a base example for using ``DRDMannTurb`` to first
calibrate a spectral tensor model to provided data and then generate a
synthetic turbulent wind field using that calibrated model.

Basic Spectra Fitting
---------------------

.. code-block:: python

    import drdmannturb as drdmt

    # TODO:


Basic Field Generation
---------------------

.. code-block:: python

   from drdmannturb.fluctuation_generation import FluctuationFieldGenerator

   # Create generator
   generator = FluctuationFieldGenerator(
       friction_velocity=0.5,
       reference_height=100.0,
       grid_dimensions=[1000.0, 500.0, 300.0],
       grid_levels=[6, 6, 5],
       model="DRD",
       path_to_parameters="path/to/calibrated/model.pkl"
   )

   # Generate field
   field = generator.generate()
