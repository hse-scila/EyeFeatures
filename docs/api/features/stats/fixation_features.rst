FixationFeatures
=================

.. currentmodule:: eyefeatures.features.stats

.. autoclass:: FixationFeatures
   :members:
   :exclude-members: __init__

Supported Metrics
-----------------

Features are taken from input dataframe, nothing additional is inferred
(in contrast with, for example, :ref:`SaccadeFeatures <saccade_features>`, where
saccades are inferred from input fixations)

- ``duration``: Duration in ms.
- ``vad``: Dispersion in ms².
