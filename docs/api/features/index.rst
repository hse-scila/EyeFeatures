Feature Extraction
==================

The ``features`` module provides tools for calculating eye-tracking features. There are two types
of features:

1. Scalar Features
------------------

In simple words, the module coheres all features that are functions from scanpaths to :math:`\mathbb{R}`.
This could be used in ``scikit-learn`` pipeline, giving an object (like person reading the text)
a mean saccade length.

For convenience, there is the :ref:`Extractor <extractor>` meta-class.

There are four submodules that work with that type of features and provide ``fit``/``transform``
interface.

.. toctree::
   :maxdepth: 1

   stats/index
   measures/index
   dist/index
   shift/index

Statistical Features
********************

The unified format to aggregate scanpaths. These are based on extracting features from saccades,
regressions, and fixations. On input, only fixations are expected, providing an
opportunity to identify saccades/regressions from them and
instantly calculate the desired statistical features, like minimum saccade length, or variance
of fixations diameter.

Refer to ``stats`` submodule.

Measures
********

The collection of algorithms for timeseries. 2D Scanpath (series of fixations) is treated
as 3D timeseries (third axis is time). One can find Hurst Exponent, Hilbert
Huang Transform, and different entropy-based methods helpful.

Refer to ``measures`` submodule.

Scanpath Distances
******************

The collection of algorithms that compare a pair of scanpaths. There is a wide range
of methods, starting from simple Euclidean/Hausdorff distances, and going up to
`Dynamic Time Warp <https://en.wikipedia.org/wiki/Dynamic_time_warping>`_ algorithm.

Refer to ``dist`` submodule.

Shift
*****

The submodule provides ``InstanceNormalization`` class interface, which is capable of
normalizing features across any user-defined dimensions. For example, normalize input features
inside the same (PersonID, Sex) slice. The class could be used in ``Pipeline`` to process
already extracted features (for instance, from ``Extractor`` class).

Refer to ``shift`` submodule.

2. Maps
-------

Instead of outputting a single number, functions in this module output a feature map. For example,
``get_mtf`` returns Markov Transition Field (MTF), which is a matrix, and cannot be integrated into
a classical Machine Learning pipeline in some unified format. Its usage is either in Deep Learning
networks or other specific analysis.

There are two submodules that work with that type of features.

.. toctree::
   :maxdepth: 1

   feature_maps/index
   pairwise/index

Feature Maps
************

The collection of algorithms to get various feature maps from scanpath. The user can find here
scanpath heatmaps, MTF, Recurrence Quantification Analysis (RQA) matrix, and other.

Refer to ``feature_maps`` submodule.

Pairwise Scanpath Distances
***************************

The collection of algorithms to aggregate several scanpaths. There are similarity/distance
matrix calculations, spectral/optimal leaf matrix reorderings, and more.

Refer to ``pairwise`` submodule.

Extractor
---------

.. toctree::
    :maxdepth: 1

    extractor
