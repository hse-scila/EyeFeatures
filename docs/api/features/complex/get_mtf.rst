get_mtf
=======

.. currentmodule:: eyefeatures.features.complex

.. autofunction:: get_mtf

Given time series :math:`X = \{x_1, .., x_n\}`, we define its quantile bins
as :math:`Q = \{q_1, ..., q_m\}` and assign each :math:`x_i` with the
corresponding bin :math:`q_j`, :math:`f: X \mapsto \{1, ..., m\}`. Then, we
construct an :math:`m\times m` weighted adjacency matrix :math:`W`, such that
:math:`W_{ij}` is number of transitions from :math:`q_i` to :math:`q_j`. After,
we normalize rows of :math:`W` such that :math:`\forall i \in \{1, ..., m\}: \sum_{j=1}^{m}W_{ij} = 1`,
we have :math:`W_{ij} = P(x_k \in q_i|x_{k - 1} \in q_j)`, probability of
transitioning from bin :math:`i` to bin :math:`j` in one step,
which is a definition of Markov Transitional Matrix (MTM).

The problem of MTM is that it does not capture the time domain information,
i.e. timestamps of :math:`X`. Instead, define Markov Transitional Field (MTF)
as a :math:`n \times n` matrix :math:`M`, where
:math:`M_{ij} = W_{f(x_i), f(x_j)}`, the probability of going from
bin :math:`f(x_i)` to bin :math:`f(x_j)` in :math:`k = |i - j|` steps.

Reference
*********

`Zhiguang Wang & Tim Oates (2015)
<https://www.researchgate.net/publication/282181246_Spatially_Encoding_Temporal_Correlations_to_
Classify_Temporal_Data_Using_Convolutional_Neural_Networks>`_.
Spatial Encoding Temporal Correlations to Classify Temporal Data Using Convolutional
Neural Networks. Served as a resource of MFT and GAF description.
