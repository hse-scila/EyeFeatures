WienerFilter
============

.. currentmodule:: eyefeatures.preprocessing.smoothing

.. autoclass:: WienerFilter
   :members:
   :exclude-members: __init__

Algorithm
*********

This filter assumes the following model of distortion:

.. math::
    g(x) = f(x) * h(x) + s(x)

where :math:`f` is true signal, :math:`h` is distortion signal,
:math:`*` is convolution operation, :math:`s` is noise,
and :math:`g` is distorted signal (observed signal). Wiener's approach considers
input signal and noise as random variables and finds estimator
:math:`\hat{f}` which minimizes the variance of :math:`\hat{f} - f`. It could be shown
that in the underlined model the minimum is achieved (in Fourier frequency domain) at:

.. math::
    \hat{F}(x) = \frac{\overline{H(x)}}{|H(x)|^2 + K}G(x)

where

* :math:`\hat{F}(x)` - Fourier-image of :math:`f`.
* :math:`H(x)` - Fourier-image of distorting function :math:`h`.
* :math:`\overline{\cdot}` - complex inverse.
* :math:`|\cdot|` - complex modulus.
* :math:`K` - approximation constant.
