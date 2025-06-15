Fixation Extraction
===================

The ``fixation_extraction`` submodule provides methods to infer fixations
from raw gazes.

While squashing gazes, we would like to extract fixations and keep the path trajectory. Thus,
the general approach is as follows: mark each gaze with :math:`0` (not a part of fixation)
or :math:`1` (a part of fixation), and then squash consecutive ones into single fixation.
Here we denote :math:`n` fixations as triplets :math:`\{(x_i, y_i, t_i)\}_{i=1}^{n}` -
:math:`x` coordinate, :math:`y` coordinate, timestamp.

There are three main algorithms to accomplish that:

.. toctree::
    :maxdepth: 1

    idt
    ivt
    ihmm

Reference
*********

Algorithms are taken from `Salvucci & Goldberg (2000)
<https://www.researchgate.net/publication/220811146_Identifying_fixations_and_saccades_in_eye-tracking_protocols>`_.
Identifying saccades and fixations in eye-tracking protocols.
