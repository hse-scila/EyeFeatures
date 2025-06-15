OverlapClustering
=================

.. currentmodule:: eyefeatures.preprocessing.aoi_extraction

.. autoclass:: OverlapClustering
   :members:
   :exclude-members: __init__


Algorithm
*********

For overlap clustering, you should provide diameters and
centers of the fixations.

1. Build the clusters. Each fixation is the particular cluster.
2. Find fixations, which are located inside other fixations, we consider this like one cluster.
3. Start to merge the clusters.
4. Find the cluster with the highest number of fixations in it (let it be the cluster <b>A</b>).
5. Find all clusters that intersect with <b>A</b> and merge them to <b>A</b>.
6. Repeat 4-5 until there are no clusters left.
