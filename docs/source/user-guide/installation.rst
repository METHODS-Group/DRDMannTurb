Installation
============

Quick Install
-------------

.. code-block:: bash

   pip install drdmannturb

Development Install
------------------

Using uv (recommended):

.. code-block:: bash

   git clone https://github.com/METHODS-Group/DRDMannTurb.git
   cd DRDMannTurb
   uv pip install -e .[docs,dev]

Using pip:

.. code-block:: bash

   git clone https://github.com/METHODS-Group/DRDMannTurb.git
   cd DRDMannTurb
   pip install -e .[docs,dev]

Verification
-----------

.. code-block:: bash

   python -c "import drdmannturb; print('Installation successful!')"
