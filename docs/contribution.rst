.. _contribution:

Contribution
============

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 and use Google-style docstrings.
2. **Testing**: Add unit tests for new features.
3. **Documentation**: Update documentation when adding new features.
4. **Issues**: Report bugs or feature requests via GitHub issues.

Development Setup
-----------------

By default, EyeFeatures package is install without the ``deep`` module. In order to
install it, you have to use ``poetry`` (see below).

First, create a new environment.

If you use ``conda``:

.. code-block:: bash

    conda create -n <name_of_environment>
    conda activate <name_of_environment>
    pip install ipykernel
    python -m ipykernel install --user --name <name_of_environment> --display-name "<name_of_environment>"

``venv``:

.. code-block:: bash

    python -m venv </path/to/new/virtual/environment>

Second, install the library.

.. code-block:: bash

    pip install eyefeatures

If you what the complete build, use ``poetry`` (in the last command, specify groups defined
in ``pyproject.toml``):

.. code-block:: bash

    git clone https://github.com/hse-scila/EyeFeatures.git
    cd eyefeatures
    pip install poetry
    poetry install --with deep dev

