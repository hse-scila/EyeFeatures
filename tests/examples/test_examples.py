import os
import runpy
import sys

import pytest

# Ensure project root is in path if running from elsewhere, though pytest usually handles this
sys.path.append(os.getcwd())


@pytest.mark.parametrize(
    "example_script",
    [
        "eyefeatures.examples.basic_fixation",
        "eyefeatures.examples.basic_features",
        "eyefeatures.examples.basic_scanpath",
        "eyefeatures.examples.advanced_pipeline",
        "eyefeatures.examples.advanced_all_features",
    ],
)
def test_example_runs(example_script):
    """
    Runs each example script using runpy to verify it executes without error.
    This acts as a high-level integration test.
    """
    print(f"\nRunning example: {example_script}")
    try:
        # run_module finds the module in sys.path and executes it
        runpy.run_module(example_script, run_name="__main__")
    except SystemExit as e:
        # Some scripts might call exit(0) on success or exit(1) on fail
        assert (
            e.code == 0 or e.code is None
        ), f"Example {example_script} exited with code {e.code}"
    except Exception as e:
        pytest.fail(f"Example {example_script} raised exception: {e}")
