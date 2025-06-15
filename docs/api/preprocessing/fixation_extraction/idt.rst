IDT
===

.. currentmodule:: eyefeatures.preprocessing.fixation_extraction

.. autoclass:: IDT
   :members:
   :exclude-members: __init__

Algorithm
*********

The algorithm uses sliding window to find consecutive gazes with
dispersion less than ``max_dispersion`` and duration more than
``min_duration``. These heuristics ensure that extracted fixations
have small variance and their duration is long enough.
