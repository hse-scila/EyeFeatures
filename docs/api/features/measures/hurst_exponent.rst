HurstExponent
=============

.. currentmodule:: eyefeatures.features.measures

.. autoclass:: HurstExponent
   :members:
   :exclude-members: __init__

* Time series is divided into segments (blocks) of equal size.
* The mean is subtracted from each segment to center the data.
* Compute the cumulative sum of the mean-adjusted data and determine the range (maximum - minimum) of the cumulative deviation.
* Calculate the standard deviation of the original segment and the ratio of the range to the standard deviation.
* The slope of he log of the block size and the log of the R/S ratio estimates the Hurst Exponent.

.. math::
    \log\frac{R}{S} = \text{HurstExponent} \cdot \log n + C

where :math:`\frac{R}{S}` is the rescaled range, :math:`n` is the block size, and :math:`C` is some constant.

Reference
*********

`Algorithm on Wiki <https://en.wikipedia.org/wiki/Hurst_exponent>`_.
