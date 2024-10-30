import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EyeFeatures'
copyright = '2024'
author = ''
version = release = '0.1.0'

# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',  # Optional, for Google/NumPy style
    'sphinx.ext.autosummary',  # Optional, for automatic summaries
    # 'sphinx_autodoc_typehints'  # Optional, for type hints in docs
]
autodoc_default_options = {
    'members': True,            # Include all members (functions, classes, etc.)
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include private members with underscore prefix
    'special-members': 'init', # Include init method in classes
}
language = 'en'
master_doc = 'index'
pygments_style = 'sphinx'
source_suffix = ['.rst', '.md']
# templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_theme_options = {
    "rightsidebar": "false"
}
html_css_files = ["horizontal_sidebar.css"]
html_static_path = ['_static']
