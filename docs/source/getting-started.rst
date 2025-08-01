.. py:currentmodule:: drdmannturb

Getting Started
===============

``DRDMannTurb`` has two major sub-modules ``spectra_fitting``
and ``fluctuation_generation``, which are described in the following sections.
We encourage users to read `the original DRD paper <https://arxiv.org/pdf/2107.11046.pdf>`_
in addition to the examples.

Spectra Fitting
===============

The ``spectra_fitting`` sub-module implements an operator regression task: using
one-point spectra and, optionally, spatial coherence function data, we train a neural network
to approximate a spectral tensor model -- we call these "Deep Rapid Distortion (DRD) models,"
since they are based on rapid distortion models and generalize the `Mann model
<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/spatial-structure-of-neutral-atmospheric-surfacelayer-turbulence/ACFE1EA8C45763481CBEB193B314E2EB>`_,
which admits three parameters: the Kolmogorov constant multiplied by the
two-thirds power law for the rate of viscous dissipation, a turbulence length scale, and a
non-dimensional time-scale related to the eddy-lifetime function.
``DRDMannTurb`` aims to enable users to fit DRD models implementing a variety of physics
while minimizing the complexity of generating synthetic turbulence fields.

The primary objects at play in the ``spectra_fitting`` sub-module are ``CalibrationProblem``,
``CustomDataLoader``, and the ``SpectralTensorModel`` super-class. One begins by constructing
a ``CustomDataLoader`` and a ``SpectralTensorModel``, both of which are passed to
a ``CalibrationProblem``. ``CalibrationProblem.calibrate()`` implements the training loop
and takes several parameters related to the training process as arguments.

Field Generation
================

The ``fluctuation_generation`` sub-module implements the "Mann method" for generating synthetic
turbulence fields with a domain-decomposition technique which constructs the field iteratively
with overlapping blocks which are then stitched together.

Installation
============

For users, pre-compiled wheels for the package are available via
``pip install drdmannturb``.

Alternatively, the rest of this section will guide you through installing
``DRDMannTurb`` and its dependencies from source. We suggest using
`uv<https://docs.astral.sh/uv/>`_ to install the package.

#.  Clone ``DRDMannTurb`` from its `GitHub repository
    <https://github.com/METHODS-Group/DRDMannTurb>`_. Enter the directory and check out
    the branch of your choice.

    .. code-block:: shell

        git clone https://github.com/METHODS-Group/DRDMannTurb.git
        cd DRDMannTurb
        git checkout main

#.  Install ``DRDMannTurb`` and its dependencies. We recommend using ``uv`` for
    faster dependency resolution, but ``pip`` will work as well.

    **Using uv (recommended):**

    .. code-block:: shell

        # Install uv if you haven't already
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Install the package in editable mode with all dependencies
        uv pip install -e .

    **Using pip (alternative):**

    .. code-block:: shell

        # Install the package in editable mode
        pip install -e .

#.  Verify the installation by running the test suite.

    .. code-block:: shell

        # Using uv
        uv run pytest test/unit_tests/ -v

        # Using pip
        python -m pytest test/unit_tests/ -v
