get_gaf
=======

.. currentmodule:: eyefeatures.features.complex

.. autofunction:: get_gaf

Given time series :math:`X = \{x_1, ..., x_n\}` of :math:`n` real-valued observations,
GAF is build using the following procedure:

1. Series is rescaled to :math:`[0, 1]` using min-max scaling (or assumed to be
already scaled, look at ``scale`` parameter of ``get_gaf`` method):

.. math::
    \tilde{x}_i = \frac{x_i - \min X}{\max X - \min X}

2. Scaled time series :math:`\tilde{X}` is converted to polar coordinates
using one of two methods (:math:`t_i` is corresponding timestamp):

    * Trigonometric formula (``to_polar='regular'``)

    .. math::
        r_i = \sqrt{\tilde{x}_i^2 + t_i^2}, \ \phi_i = arctan\left(\frac{t_i}{\tilde{x}_i}\right)

    * Cosine formula (``to_polar='cosine'``)

    .. math::
        r_i = \frac{t_i}{n}, \ \phi_i = arccos(\tilde{x}_i)

3. Then matrix :math:`M` is constructed, again with one of two ways:
    * Cosine of sum (``field_type='sum'``)

    .. math::
        (M)_{ij} = \cos(\phi_i + \phi_j)

    * Sine of difference (`field_type='difference'`)

    .. math::
        (M)_{ij} = \sin(\phi_i - \phi_j)

Reference
*********

`Zhiguang Wang & Tim Oates (2015)
<https://www.researchgate.net/publication/282181246_Spatially_Encoding_Temporal_Correlations_to_
Classify_Temporal_Data_Using_Convolutional_Neural_Networks>`_.
Spatial Encoding Temporal Correlations to Classify Temporal Data Using Convolutional
Neural Networks. Served as a resource of MFT and GAF description.
