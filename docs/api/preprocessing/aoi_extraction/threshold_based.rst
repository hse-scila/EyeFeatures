ThresholdBased
==============

.. currentmodule:: eyefeatures.preprocessing.aoi_extraction

.. autoclass:: ThresholdBased
   :members:
   :exclude-members: __init__


Algorithm
*********

1. Split the graph into a grid and compute density for each sector via
   gaussian kernel density estimation.

2. Pre-threshold it.

3. Find the local maxima. Every maximum is a center of area of interest.

4. Start to define AOI for each fixation.

5. If ```algorithm_type``` is default, then it starts KMeans. Otherwise,
   if ```algorithm_type='basic'```, then it starts to search for the for the
   nearest fixation with AOI and assigns this label to the target fixation.

6. Repeat for all instances.
