.. _get_hilbert_curve:

get_hilbert_curve
=================

.. currentmodule:: eyefeatures.features.complex

.. autofunction:: get_hilbert_curve

Hilbert Curve is a way to describe 2D space using 1D object. It provides a
bijection (unique mapping) of 2D plane on a 1D line, constructed as a limit of
piecewise linear curves, :math:`n`-th curve having a length of :math:`2^n - \frac{1}{2^n}`.
On practice, we fix some :math:`p` and consider :math:`p`-th Hilbert Curve and
map each :math:`(x,y)` point to the closest point on it, where higher :math:`p`
means that more space is filled with a curve which results in more accurate encoding.

Reference
*********

`Hilbert Curve on Wiki <https://en.wikipedia.org/wiki/Hilbert_curve>`_.

`Hilbert Curve encoding algorithm <https://people.math.sc.edu/Burkardt/py_src/hilbert_curve/hilbert_curve.py>`_.
