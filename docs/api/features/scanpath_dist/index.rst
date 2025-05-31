Scanpath Distances
==================

.. currentmodule:: eyefeatures.features.scanpath_dist

The ``scanpath_dist`` module provides common methods to measure distance between
scanpaths, treating them as different types of timeseries. The main idea is to
calculate expected scanpath and compare it with the input scanpath.

Base Distance Transformer
-------------------------
.. autoclass:: DistanceTransformer
   :members:
   :exclude-members: __init__, _calc_feats, _check_params

Distance Transformers
---------------------

All transformers in this list own ``fit``/``transform`` methods. As well as
class instances, there are functions in section below that provide the same
functionality (i.e. calculate distances between scanpaths).

.. toctree::
    :maxdepth: 1

    simple_distances
    euc_dist
    hau_dist
    dtw_dist
    scan_match_dist
    mannan_dist
    eye_analysis_dist
    df_dist
    tde_dist
    multi_match_dist

Distance Functions
------------------

The functions in this list are used in corresponding transformers.

.. toctree::
    :maxdepth: 1

    calc_dfr_dist
    calc_dtw_dist
    calc_euc_dist
    calc_eye_dist
    calc_hau_dist
    calc_man_dist
    calc_mm_features
    calc_scan_match_dist
    calc_tde_dist
