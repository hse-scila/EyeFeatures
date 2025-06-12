RQAMeasures
===========

.. currentmodule:: eyefeatures.features.measures

.. autoclass:: RQAMeasures
   :members:
   :exclude-members: __init__

The metrics calculated include Recurrence (REC), Determinism (DET), Laminarity (LAM),
and Center of Recurrence Mass (CORM). These measures help to quantify the complexity and
structure of the recurrence patterns within the data. In this example we use a default
euclidean metric as ``metric``. Parameters ``rho`` and ``min_length`` correspond for RQA matrix
threshold radius and threshold length of its diagonal. In ``measures`` we specify
the required features to calculate.

Recurrence matrix :math:`R` is defined as
:math:`R_{ij} = \mathbb{I}\left\{d(x_i, x_j) \leq \rho \right\}`:

* Reccurence Rate counts the total number of recurrence points above the main diagonal of :math:`R`:

.. math::
    \text{REC} = \frac{2}{n(n-1)} \sum_{i=1}^n \sum_{j=i+1}^n R_{ij}

* Determinism measures the percentage of recurrence points forming diagonal
  lines of length at least $L_{min}$:

.. math::
    \text{DET} = \frac{100 \cdot \sum_{l \geq L_{min}} l \cdot P(l)}{\sum_{i=1}^n \sum_{j=i+1}^n R_{ij}},

.. math::
    \text{ where } L_{min} - \text{ minimum line length}, \, P(l) - \text{probability of diagonal lines of length } l

* Liminarity measures the percentage of recurrence points forming
  vertical or horizontal lines of length at least :math:`L_{min}`:

.. math::
    \text{LAM} = \frac{50 \left( \sum_{\text{HL}} \text{HL} + \sum_{\text{VL}} \text{VL}\right)}{\sum_{i=1}^n \sum_{j=i+1}^n R_{ij}},

where :math:`HL` and :math:`VL` represents the sums of horizontal and vertical lines
of length at least :math:`L_{min}`.

* Center of Recurrence Mass measures the weighted average of the distances between recurrence points,
  emphasizing the central tendency of recurrences in the matrix:

.. math::
    \text{CORM} = \frac{100 \cdot \sum_{i=1}^{n-1} \sum_{j=i+1}^n (j-i) R_{ij}}{(n-1) \cdot \sum_{i=1}^n \sum_{j=i+1}^n R_{ij}}

Reference
*********

Anderson, N. C., Bischof, W. F., Laidlaw, K. E. W., Foulsham, T.,
Kingstone, A., & Cristino, F. (2013). Recurrence quantification analysis
of eye movements. Behavior Research Methods, 45(3),
842–856. https://doi.org/10.3758/s13428-012-0299-5
