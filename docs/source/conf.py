# noqa
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"

project = "DRDMannTurb"
copyright = "Alexey Izmailov, Matthew Meeker"
author = "Alexey Izmailov, Matthew Meeker"
release = "0.1"

nbsphinx_prolog = r"""
.. raw:: html

    <script src='http://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js'></script>
    <script>require=requirejs;</script>


"""

html_js_files = [
    "require.min.js",  # Add to your _static
    "custom.js",
]


# declare examples directory for sphinx-gallery scraper
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "run_stale_examples": False,
    "filename_pattern": "0*",
    "example_extensions": {".py"},
}

extensions = [
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "enum_tools.autoenum",
    "sphinxcontrib.mermaid",  # https://github.com/mgaitan/sphinxcontrib-mermaid#building-pdfs-on-readthedocsio
]

mermaid_params = ["-p" "puppeteer-config.json"]
mermaid_d3_zoom = True
myst_fence_as_directive = ["mermaid"]

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
autodoc_typehints = "both"
autoclass_content = "both"
napoleon_use_param = True
napoleon_use_rtype = False

nbsphinx_execute = "always"  # make sure that notebooks are always executed
nbsphinx_requirejs_path = ""  # for mermaid js to work

# NOTE: these settings result in warnings for plots from the traitlets module.
# nbsphinx_execute_arguments = [
# "--InlineBackend.figure_formats={'svg', 'pdf'}",
# "--InlineBackend.rc={'figure.dpi': 96}",
# ]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_sidebars: dict[str, list[str]] = {"**": []}
