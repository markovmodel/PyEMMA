r"""

Data handling and IO
--------------------
.. autosummary::
   :toctree: generated/

   featurizer
   load
   source
   pipeline
   discretizer
   save_traj
   save_trajs

Transformations
---------------
.. autosummary::
   :toctree: generated/

   pca
   tica

Clustering Algorithms
---------------------
.. autosummary::
   :toctree: generated/

   cluster_kmeans
   cluster_regspace
   cluster_uniform_time
   assign_to_centers


"""
from .api import *

from .pipelines import Pipeline
from transform.pca import PCA
from transform.tica import TICA
from clustering.kmeans import KmeansClustering
from clustering.uniform_time import UniformTimeClustering
from clustering.regspace import RegularSpaceClustering
from clustering.assign import AssignCenters


