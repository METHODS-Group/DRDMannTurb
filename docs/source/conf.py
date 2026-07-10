# noqa
from importlib.metadata import version as _version

import plotly.io as pio

pio.renderers.default = "sphinx_gallery"

project = "DRDMannTurb"
copyright = "Alexey Izmailov, Matthew Meeker"
author = "Alexey Izmailov, Matthew Meeker"
release = _version("drdmannturb")

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
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxcontrib.mermaid",  # https://github.com/mgaitan/sphinxcontrib-mermaid#building-pdfs-on-readthedocsio
]

mermaid_params = ["-ppuppeteer-config.json"]
mermaid_d3_zoom = True
myst_fence_as_directive = ["mermaid"]

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

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_sidebars: dict[str, list[str]] = {"**": []}
