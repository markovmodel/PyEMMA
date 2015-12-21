from sklearn import datasets
iris = datasets.load_iris().data
from matplotlib  import pyplot as plt
import matplotlib
#print matplotlib.get_backend()
#raise
from pyemma.coordinates import pca,cluster_mini_batch_kmeans as cluster_kmeans
trans = pca(iris, dim=2).get_output()[0]
cl = cluster_kmeans(trans, k=2)
centers = cl.clustercenters
plt.scatter(trans[:,0], trans[:,1])
plt.scatter(centers[:, 0], centers[:,1], color='red', label='x')
plt.show()
#print cl.clustercenters
