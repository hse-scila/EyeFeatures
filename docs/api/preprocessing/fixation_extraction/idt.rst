IDT
===

.. currentmodule:: eyefeatures.preprocessing.fixation_extraction

.. autoclass:: IDT
   :members:
   :exclude-members: __init__

Algorithm
*********

In order to identify fixations, IDT finds sequences of gazes that are not
too short, not too long, and are placed close enough to each other. For example,
user may define these restrictions to have fixations at
least :math:`200` ms, at most :math:`600` ms, and
with dispersion less than :math:`0.5` units. Algorithm uses sliding window to
first select gazes satisfying maximum duration constraint. Then,
it uses binary search to find the widest window
that satisfies the dispersion constraints inside the sliding window.
Finally, minimum dispersion requirement is checked, and if fulfilled,
the sequence is marked as a single fixation.

The algorithm guarantees that no two consecutive fixations could be merged together.
In other words, if their gaze sequences are considered as a single
continuous sequence, it would not satisfy the constraints on duration and time.

Implementation Details
**********************

Formal description is as follows. Consider set of :math:`n` gazes
:math:`\{g_i\}_{i=1}^{n}`, where each gaze
:math:`g_i = (x_i, y_i)` is a point in :math:`\mathbb{R}^2`. Also, the gazes
have corresponding timestamps :math:`\{t_i\}_{i=1}^{n}`.

1. Sequence on range :math:`[l, r]` satisfies duration thresholds :math:`T_{min}`
and :math:`T_{max}` if

.. math::
    T_{min} \leq t_r - t_l \leq T_{max}

2. Sequence on range :math:`[l, r]` satisfies dispersion threshold :math:`D` if

.. math::
    \max_{(i, j) \in [l, r]^2, i < j} d(g_i, g_j) \leq D

That is, the definition of dispersion of a set of gaze points
we use is maximum pairwise distance, where the distance metric is
arbitrary. However, exact calculation of dispersion requires
knowledge of distances between any two gaze points. The number of
such pairs is :math:`O(n^2)`, resulting in a pure quadratic complexity.
Such approach is not feasible for a scanpath of
length :math:`n \geq 10^4` in Python.

Since computing dispersion is required on each iteration of binary search,
the algorithm uses an approximation of dispersion to optimize performance.
Instead of checking all pairwise distances, a cloud of points is inscribed into a
:math:`d`-dimensional cube (:math:`d = 2` in our case, i.e. plane) and
the diagonal length is taken as the corresponding dispersion value. This
approach requires to answer RMQ-s (Range Min/Max Queries). Such queries
are answered with four precomputed sparse tables, two for each coordinate
(:math:`x` and :math:`y`). Their construction requires :math:`O(n\log n)`
time.

Complexity
**********

The EyeFeatures implementation requires :math:`O(n\log n)` memory,
having an overall complexity of :math:`O(n W_{avg} \log n)`, where
:math:`W_{avg}` is average size of sliding window per gaze point. In worst case,
if `max_dispersion`:math:`<` distance between any two points
and duration of whole input gaze sequence is :math:`<` `max_duration`,
:math:`W_{avg}` becomes equal to :math:`O(n)`, leveraging the complexity
of the algorithm to :math:`O(n^2\log n)`. However, this scenario is impractical
and on average :math:`W_{avg} << n`, resulting in reduction to :math:`O(n\log n)`.
Smaller asymptotic could not be achieved if sparse tables are used, since their
construction requires :math:`O(n\log n)` memory and time.
