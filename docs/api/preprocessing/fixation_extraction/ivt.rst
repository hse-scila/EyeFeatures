IVT
===

.. currentmodule:: eyefeatures.preprocessing.fixation_extraction

.. autoclass:: IVT
   :members:
   :exclude-members: __init__

Algorithm
*********

Gazes that have velocity below threshold are considered to be fixations,
since high velocities are attributes of saccades. If :math:`d` is
some metric in :math:`\mathbb{R}^2` and :math:`a` is
fixation-classification decision function, then, for a single fixation:

.. math::
    a(i) = \left[\frac{d((x_i, y_i), (x_{i + 1}, y_{i + 1}))}{t_{i + 1} - t_i} \leq T\right]
