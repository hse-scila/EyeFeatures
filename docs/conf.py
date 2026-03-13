import os
import sys

sys.path.insert(0, os.path.abspath(".."))


def setup(app):
    import importlib
    import inspect
    import os
    import pkgutil

    def generate_api_index():
        # Modules to scan
        scan_modules = [
            "eyefeatures.features",
            "eyefeatures.preprocessing",
            "eyefeatures.visualization",
            "eyefeatures.utils",
        ]

        ignored_classes = (
            "BaseTransformer",
            "BasePreprocessor",
            "BaseAOIPreprocessor",
            "BaseSmoothingPreprocessor",
            "BaseFixationPreprocessor",
            "BaseBlinkPreprocessor",
            "Types",
        )

        def is_ignored(name, obj) -> bool:
            if name.startswith("_"):
                return True
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                return True
            mod_of_obj = getattr(obj, "__module__", "")
            if not mod_of_obj.startswith("eyefeatures"):
                return True
            if name in ignored_classes:
                return True
            return False

        seen = set()
        items = []

        for module_name in scan_modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            path = getattr(module, "__path__", None)
            if path:
                for loader, name, is_pkg in pkgutil.walk_packages(
                    path, module_name + "."
                ):
                    try:
                        submodule = importlib.import_module(name)
                        for name, obj in inspect.getmembers(submodule):
                            if not is_ignored(name, obj):
                                mod_of_obj = getattr(obj, "__module__", "")
                                full_name = f"{mod_of_obj}.{name}"
                                if full_name not in seen:
                                    items.append((name, full_name))
                                    seen.add(full_name)
                    except Exception:
                        continue
            else:
                for name, obj in inspect.getmembers(module):
                    if not is_ignored(name, obj):
                        mod_of_obj = getattr(obj, "__module__", "")
                        full_name = f"{mod_of_obj}.{name}"
                        if full_name not in seen:
                            items.append((name, full_name))
                            seen.add(full_name)

        items.sort(key=lambda x: x[1].lower())

        output_path = os.path.join(os.path.dirname(__file__), "all_methods.rst")
        with open(output_path, "w") as f:
            f.write("All Methods and Classes\n")
            f.write("=======================\n\n")
            f.write(
                "This is an alphabetical list of all public transformers and functions in the ``eyefeatures`` library.\n\n"
            )
            f.write(".. autosummary::\n")
            f.write("   :nosignatures:\n\n")
            for name, full_name in items:
                f.write(f"   {full_name}\n")

    generate_api_index()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EyeFeatures"
copyright = "2025"
author = "Vagiz Daudov"
version = release = "2.0.0"

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
