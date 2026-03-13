MannanDist
==========

.. currentmodule:: eyefeatures.features.scanpath_dist

.. autoclass:: MannanDist
   :members:
   :exclude-members: __init__

Mannan distance is somewhat a more complex version of EyeDist since it considers the weighted distance:

.. math::
    \text{MAN}(p, q) = \frac{1}{4 \cdot n \cdot m} \left(m \cdot \sum_{i=1}^n \min_{1 \leq j \leq m} ||p_i - q_j||_2^2 + n \cdot \sum_{j=1}^m \min_{1 \leq i \leq n} ||q_j - p_i||_2^2  \right)
