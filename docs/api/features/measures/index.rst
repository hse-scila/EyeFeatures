Measures
========

.. currentmodule:: eyefeatures.features.measures

The ``measures`` submodule provides methods to infer some characteristics from scanpaths,
treating them as either 2D, 3D, or 4D timeseries.

Base Measure Transformer
------------------------
.. autoclass:: MeasureTransformer
   :members:
   :exclude-members: __init__, _calc_feats, _check_params


Measure Transformers
--------------------

Common measures
***************

.. toctree::
    :maxdepth: 1

    hurst_exponent
    lyapunov_exponent
    fractal_dimension
    correlation_dimension
    rqa_measures
    saccade_unlikelihood
    hht_features

Entropy-based
*************

.. toctree::
    :maxdepth: 1

    shannon_entropy
    spectral_entropy
    fuzzy_entropy
    sample_entropy
    incremental_entropy
    gridded_distribution_entropy
    phase_entropy
