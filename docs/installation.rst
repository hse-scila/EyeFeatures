.. _installation:

Installation
============

The library can be installed via pip:

.. code-block:: bash

    pip install eyefeatures

Requirements
------------
- Python 3.10, 3.11, 3.12, or 3.13
- See ``pyproject.toml`` for full dependency list

Optional: Deep Learning Module
------------------------------

The ``deep`` module (neural network models) requires PyTorch and is not
installed by default. To include it:

.. code-block:: bash

    pip install poetry
    git clone https://github.com/hse-scila/EyeFeatures.git
    cd EyeFeatures
    poetry install --with deep

Main Dependencies
-----------------
- ``numpy``
- ``pandas``
- ``scikit-learn``
- ``scipy``
- ``matplotlib``
- ``torch`` (optional, for deep module)

Full dependency list available in
`pyproject.toml <https://github.com/hse-scila/EyeFeatures/blob/main/pyproject.toml>`_.
