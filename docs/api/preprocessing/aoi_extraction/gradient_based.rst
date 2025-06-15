GradientBased
=============

.. currentmodule:: eyefeatures.preprocessing.aoi_extraction

.. autoclass:: GradientBased
   :members:
   :exclude-members: __init__

Algorithm
*********

1. Split the graph into a grid and compute density for each
   sector via gaussian kernel density estimation.
2. Pre-threshold it.
3. Find the local maxima. Every maximum is a center of area of interest.
4. Compute the gradient and magnitude.
5. Add all 8 points around of each maximum to the queue.
6. Get a point from the queue. Search among 8 points around the point with the
   greatest magnitude, which has AOI. Assign this label to the target point.
7. Add all 8 points with around the target point to the queue.
8. Repeat 5-8 until all points will get the AOI label.
