.. _ref-coordinates:

Coordinates API
===============
The *coordinates* package contains tools to select features from MD-trajectories.
It also assigns them to a discrete state space, which will be later used in Markov
modeling.

It supports reading from MD-trajectories, comma separated value ASCII files and 
NumPy arrays. The discretized trajectories are being stored as NumPy arrays of
integers.

.. automodule:: pyemma.coordinates

.. toctree::
   :maxdepth: 1

Coordinate classes
------------------

.. toctree::
   :maxdepth: 1

   coordinates.data
   coordinates.pipelines
   coordinates.transform
   coordinates.clustering
