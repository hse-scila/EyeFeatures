EucDist
=======

.. currentmodule:: eyefeatures.features.scanpath_dist

.. autoclass:: EucDist
   :members:
   :exclude-members: __init__

Euclidean distance is simply the sum of pairwise distances of two sequences at each timestamp:

.. math::
   \text{EUC}(p, q) = \sum_{i=1}^n ||p_i - q_i||_2

where :math:`p` and :math:`q` are aligned.
