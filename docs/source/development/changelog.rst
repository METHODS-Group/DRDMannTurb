Changelog
=========

Version 1.1
-----------
* Significant refactoring of ``spectra_fitting`` with a new API based on building "models" from components
    * See ``spectral_tensor_models.py`` for the new "models" API
    * A new, standardized data loader API using ``polars.DataFrame``
    * Significant changes to ``CalibrationProblem`` interface (most things moved out of the constructor
      to wherever they are first required)
    * Removed all non-/re-dimensionalizing constants from the ``spectra_fitting`` backend --
      now, users are expected to handle the non-dimensionalization of the inputs and the re-dimensionalization
      at inference time.
* Updated tooling for development and documentation
    * Primarily, we now rely on ``uv`` and ``ruff``
    * Massively improved testing suite
    * Creation of a ``Makefile`` for common development tasks
    * Substantial reworking of the documentation

Version 1.1-beta.0
------------------

* Initial beta release
* Modern Python packaging with uv
* With one-point spectra data, one can now provide one of the following:
    * All auto-spectra components (uu, vv, ww)
    * All auto- and uw cross-spectra components (uu, vv, ww, uw)
    * All auto- and all cross-spectra components (uu, vv, ww, uw, vw, uv)
