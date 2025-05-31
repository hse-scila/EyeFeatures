IHMM
====

.. currentmodule:: eyefeatures.preprocessing.fixation_extraction

.. autoclass:: IHMM
   :members:
   :exclude-members: __init__

Algorithm
*********

The algorithm finds a sequence of fixations that maximizes the
log probability of observing given velocities of gazes under conditions
of Hidden Markov Model. More formally, denote velocity of
:math:`i`-th gaze as

.. math::
    v_i = \frac{d((x_i, y_i), (x_{i + 1}, y_{i + 1}))}{t_{i + 1} - t_i}

This is observed process, while hidden process is sequence :math:`\{s_i\}_{i=1}^{n}`
of zeros and ones, as mentioned in previous section. Given some fixed prior distribution
of velocities (Gaussian is taken as empirical rule) and transition matrix, then,
under assumption of Markov process, probability is maximized in greedy manner.

The process is called Markov process if

.. math::
    P(s_i = b|v_{i - 1}, ..., v_1) = P(s_i = b|v_{i - 1}, ..., v_{i - k})

for some fixed :math:`k \geq 1`.
