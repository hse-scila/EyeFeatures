import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EyeFeatures"
copyright = "2025"
author = ""
version = release = "1.0.1"

# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "m2r2",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# MathJax configuration (for HTML)
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "packages": {"base": True, "ams": True, "amssymb": True},  # Essential packages
        "inlineMath": [["\\(", "\\)"]],  # Single backslashes
        "displayMath": [["\\[", "\\]"]],  # Single backslashes
    }
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__,_check_init,_validate_data",
    "show-inheritance": True,
}

autodoc_typehints = "description"

language = "en"
master_doc = "index"
pygments_style = "sphinx"
source_suffix = [".rst", ".md"]
# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {
#     "rightsidebar": "false"
# }
# html_css_files = ["horizontal_sidebar.css"]
# html_static_path = ['_static']
