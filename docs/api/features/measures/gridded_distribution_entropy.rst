GriddedDistributionEntropy
==========================

.. currentmodule:: eyefeatures.features.measures

.. autoclass:: GriddedDistributionEntropy
   :members:
   :exclude-members: __init__

* Given a set of 2D points :math:`\left\{(x_i, y_i) \right\}_{i=1}^N` algorithm partitions
  the space into a grid consisting of :math:`g \times g` cells.

* Define the edges of the cells for each dimension: :math:`\text{Edges}_x = \left\{x_0, \dots, x_g \right\},`
  :math:`\text{Edges}_y = \left\{y_0, \dots, y_g \right\}`.

* Then, each bin is basically a
  :math:`B_{jk} = \left\{ (x, y): x_{j-1} \leq x < x_j, \, y_{k-1} \leq y < y_k \right\}`.

* Construct a **multi-dimensional histogram** :math:`H` where each element
  :math:`H_{jk}` represents the count of data points falling to the :math:`B_{jk}`:

.. math::
    H_{jk} = \sum_{i=1}^N \mathbb{I}\left\{(x_i, y_i) \in B_{jk} \right\}

* Normalize the histogram to obtain a probability distribution
  :math:`P \sim P_{jk} = \frac{H_{jk}}{N}` and calculate its entropy.

.. math::
    S = -\sum_{i=1}^g\sum_{j=1}^g P_{jk}\log(P_{jk})

Reference
*********

Melnyk, K., Friedman, L., & Komogortsev, O. V. (2024). What can entropy metrics
tell us about the characteristics of ocular fixation trajectories? PLoS ONE, 19(1),
e0291823. https://doi.org/10.1371/journal.pone.0291823.
