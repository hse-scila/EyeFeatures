AOI Extraction
==============

The ``aoi_extraction`` submodule has transformers to identify AOI areas for input
fixations. As well as standalone transformers, there is a convenient
meta transformer that allows to specify all desired methods and use
several AOI extraction methods at the same time.


.. toctree::
    :maxdepth: 1

    shape_based
    threshold_based
    gradient_based
    overlap_clustering

AOI Matcher
***********

There is also a class that works with already defined Areas of Interest. It can merge
regions together if they are strongly intersected. Formally, it minimizes
Shannon Entropy. It is strongly recommended to use it as post-processor after
any of above transformers.

.. toctree::
    :maxdepth: 1

    aoi_matcher

Reference
*********

The algorithms are taken from `Wolfgang Fuhl. “Image-based extraction of
eye features for robust eye tracking”. 2019.
<https://www.hci.uni-tuebingen.de/assets/pdf/publications/WF042019.pdf>`_
