# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Test Project'
author = 'The Authors'
version = release = '0.1.0'

# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = []
extensions = [
    'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional, for Google/NumPy style
    'sphinx.ext.autosummary',  # Optional, for automatic summaries
    'sphinx_autodoc_typehints'  # Optional, for type hints in docs
]
autodoc_default_options = {
    'members': True,            # Include all members (functions, classes, etc.)
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include private members with underscore prefix
    'special-members': 'init', # Include init method in classes
}
language = 'en'
# master_doc = 'index'
pygments_style = 'sphinx'
source_suffix = ['.rst', '.md']
# templates_path = ['_templates']

# -- Options for HTML output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
