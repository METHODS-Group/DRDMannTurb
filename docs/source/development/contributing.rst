
Contributing to DRDMannTurb
===========================

Development Setup
----------------

1. Clone the repository
2. Install with uv: ``uv pip install -e ".[dev]"``
3. Install ``pre-commit`` and hooks: ``uv run pre-commit install``
3. Run tests: ``uv run pytest test/``
4. Run linting: ``uv run ruff check drdmannturb/``

Code Style
----------

We use ruff for linting and formatting:

.. code-block:: bash

   uv run ruff check drdmannturb/
   uv run ruff format drdmannturb/

Testing
--------

.. code-block:: bash

   # Run all tests
   uv run pytest test/

   # Run specific test groups
   uv run pytest test/unit_tests/
   uv run pytest test/integration_tests/

   # Run with coverage
   uv run pytest test/ --cov=drdmannturb
