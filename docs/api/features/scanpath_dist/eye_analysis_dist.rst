EyeAnalysisDist
===============

.. currentmodule:: eyefeatures.features.scanpath_dist

.. autoclass:: EyeAnalysisDist
   :members:
   :exclude-members: __init__

EyeDist distance is calculated as follows:

.. math::
    \text{EYE}(p, q) = \frac{1}{\max\{n, m\}} \left(\sum_{i=1}^n \min_{1 \leq j \leq m} ||p_i - q_j||_2^2 + \sum_{j=1}^m \min_{1 \leq i \leq n} ||q_j - p_i||_2^2\right)
