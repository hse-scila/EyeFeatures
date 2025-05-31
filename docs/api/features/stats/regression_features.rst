RegressionFeatures
===================

Regressions are defined as a subclass of saccades, pointing in certain direction.

.. currentmodule:: eyefeatures.features.stats

.. autoclass:: RegressionFeatures
   :members:
   :exclude-members: __init__

Supported Metrics
-----------------
- ``length``: Amplitude in pixels.

.. math::
    \text{Length(Saccade}_i\text{)} = ||\text{Fixation}_{i+1} - \text{Fixation}_{i} ||_{2}

- ``speed``: Velocity in pixels/ms.

.. math::
    \text{Speed(Saccade}_i\text{)} = \frac{\text{Length(Saccade}_i\text{)}}{\text{Time}_{i+1} - \text{Time}_{i}}

- ``acceleration``: Acceleration in pixels/ms².

.. math::
    \text{Acceleration(Saccade}_i\text{)} = \frac{1}{2} \frac{\text{Speed(Saccade}_i\text{)} }{\text{Time}_{i+1} - \text{Time}_{i}}

- ``mask``: Transition boolean mask.

.. math::
    \text{Mask(Saccade}_i\text{)} = \mathbb{I}[\text{Fixation}_{i}\in\text{ some regression}]
