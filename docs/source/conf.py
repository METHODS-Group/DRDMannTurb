# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DRDMannTurb"
copyright = "2023, Matthew Meeker, Alexey Izmailov"
author = "Alexey Izmailov, Matthew Meeker"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

autodoc_mock_imports = ["torch"]
# autodoc_preserve_defaults = False
# auto_docstring_signature = False

autosummary_generate = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ArrayLike": ":term:`array-like`",
    "NDArray": "~numpy.ndarray",
    "NDArray[np.float64]": "~numpy.ndarray",
    "NDArray[Union[np.float64, np.int64]]": "~numpy.ndarray",
}
napoleon_use_admonition_for_notes = True
autodoc_typehints = "none"
napoleon_use_param = True
napoleon_use_rtype = False

nbsphinx_execute = "always"  # make sure that notebooks are always executed

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
