import os
import runpy
import sys

import pytest

# Ensure project root is in path
sys.path.append(os.getcwd())


@pytest.mark.parametrize(
    "example_file",
    [
        "basic_fixation.py",
        "basic_features.py",
        "basic_scanpath.py",
        "advanced_pipeline.py",
        "advanced_all_features.py",
    ],
)
def test_example_runs(example_file):
    """
    Runs each example script using runpy to verify it executes without error.
    This acts as a high-level integration test.
    """
    # Assuming tests are run from project root, examples are in ./examples/
    example_path = os.path.join("examples", example_file)
    if not os.path.exists(example_path):
        pytest.fail(f"Example file not found: {example_path}")

    print(f"\nRunning example: {example_path}")
    try:
        # run_path executes the file at the given path
        runpy.run_path(example_path, run_name="__main__")
    except SystemExit as e:
        # Some scripts might call exit(0) on success or exit(1) on fail
        assert (
            e.code == 0 or e.code is None
        ), f"Example {example_file} exited with code {e.code}"
    except Exception as e:
        pytest.fail(f"Example {example_file} raised exception: {e}")
