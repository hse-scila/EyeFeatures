Statistical Features
====================

.. currentmodule:: eyefeatures.features.stats

The ``stats`` submodule provides statistical analysis of eye movement components.

Base Feature Transformer
------------------------
.. autoclass:: StatsTransformer
   :members:
   :exclude-members: __init__, _calc_feats, _check_params

Feature Transformers
--------------------

.. toctree::
   :maxdepth: 1

   saccade_features
   micro_saccade_features
   regression_features
   fixation_features
