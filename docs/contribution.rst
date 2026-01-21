.. _contribution:

Contribution
============

We welcome contributions! This guide explains the development workflow.

Development Setup
-----------------

First, clone the repository and set up your environment:

.. code-block:: bash

    git clone https://github.com/hse-scila/EyeFeatures.git
    cd EyeFeatures

Create a virtual environment with Python 3.10, 3.11, or 3.12:

.. code-block:: bash

    # Using venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows

    # Or using conda
    conda create -n eyefeatures python=3.12
    conda activate eyefeatures

Install the library with development dependencies:

.. code-block:: bash

    pip install poetry
    poetry install --with dev,test,deep

Set up pre-commit hooks:

.. code-block:: bash

    pre-commit install


Development Workflow
--------------------

1. **Create a feature branch**

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Write your code**

   - Follow PEP 8 style guidelines
   - Use Google-style docstrings
   - Add type hints to function signatures

3. **Run pre-commit checks**

   Before committing, ensure your code passes all linting checks:

   .. code-block:: bash

       pre-commit run --all-files

   This runs:

   - **black**: Code formatting.
   - **ruff**: Fast linting with auto-fixes.

   If any tool modifies files, re-run to confirm they pass.

4. **Run tests**

   Ensure all tests pass with at least 80% coverage:

   .. code-block:: bash

       pytest --cov=eyefeatures --cov-report=term-missing

   Expected: **All tests passing** and **≥80% coverage**.

5. **Commit and push**

   .. code-block:: bash

       git add .
       git commit -m "feat: your descriptive commit message"
       git push origin feature/your-feature-name

6. **Open a Pull Request**

   Create a PR on GitHub. CI will automatically run linting and tests.
   Address any review feedback before merging.


Code Quality Standards
----------------------

- **Test coverage**: Minimum 80% required
- **Line length**: 88 characters maximum
- **Docstrings**: Required for all public functions/classes
- **Type hints**: Strongly encouraged


Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

    # Install docs dependencies
    poetry install --with docs

    # Build HTML docs
    cd docs
    poetry run make html

    # View in browser
    xdg-open _build/html/index.html  # Linux
    # open _build/html/index.html  # macOS

Documentation is built with Sphinx and uses the ReadTheDocs theme.
