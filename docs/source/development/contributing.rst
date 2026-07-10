
Contributing to DRDMannTurb
===========================

Development Setup
-----------------

1. Clone the repository
2. Install the environment with uv: ``uv sync``
3. Install ``pre-commit`` and hooks: ``uv run pre-commit install``
4. Run tests: ``uv run pytest test/``
5. Run linting: ``uv run ruff check drdmannturb/``

The ``Makefile`` bundles the common tasks: ``make check`` runs lint, typecheck, and unit tests.

Code Style
----------

We use ruff for linting and formatting:

.. code-block:: bash

   uv run ruff check drdmannturb/
   uv run ruff format drdmannturb/

Testing
-------

.. code-block:: bash

   # Run all tests
   uv run pytest test/

   # Run specific test groups
   uv run pytest test/unit_tests/
   uv run pytest test/integration_tests/

   # Include slow integration tests
   uv run pytest test/ --runslow

   # Run with coverage
   uv run pytest test/ --cov=drdmannturb
