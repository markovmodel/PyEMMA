'''
Created on Nov 16, 2013

@author: noe
'''

import os

from emma2.coordinates.transform import *

# find EMMA
emma_path = "/Users/noe/data/software/emma1.4.1/bin/"

def create_custom_transform(name, dir_input, dir_output, custom_transform, transform_only_new=False, output_extension=None):
    return filetransform.FileTransform(name, custom_transform, dir_input, dir_output, filepattern="*", 
                         transform_only_new=transform_only_new,
                         output_extension=output_extension)

def create_crd2dist_transform(dir_input, dir_output, set1, set2=None, output_extension=None):
    crd2dist = transform_crd2dist.Transform_crd2dist(set1, set2)
    ft = filetransform.FileTransform("Distance computation", crd2dist, dir_input, dir_output, filepattern="*", transform_only_new=True, output_extension=output_extension)
    return ft 

def create_tica_transform(dir_input, dir_tica, dir_output, lag, ndim, output_extension=None):
    tica = transform_tica.Transform_TICA(lag, ndim, dir_tica=dir_tica, output_extension=output_extension)
    ft = filetransform.FileTransform("TICA", tica, dir_input, dir_output, filepattern="*", transform_only_new=False, output_extension=output_extension)
    return ft 

def create_clustering_regspace(dir_input, dir_work, dir_output, dmin, cluster_skipframes = 1):
    cluster = transform_clusterassign.Transform_ClusterAssign(dir_work+"/clustercenters.dat", 
                                                skipframes = cluster_skipframes,
                                                filepattern="*", 
                                                parameters={'algorithm': 'regularspatial', 'dmin': dmin, 'metric': 'euclidean'})
    ft = filetransform.FileTransform("Regular-space clustering", cluster, dir_input, dir_output, filepattern="*", transform_only_new=False)
    return ft 

def create_clustering_kmeans(dir_input, dir_work, dir_output, ncluster, cluster_skipframes = 1):
    return transform_clusterassign.Transform_ClusterAssign("k-means clustering", dir_input, dir_work+"/clustercenters.dat", dir_output, 
                 skipframes = cluster_skipframes,
                 filepattern="*", 
                 parameters={'algorithm': 'kmeans', 'clustercenters': ncluster, 'metric': 'euclidean'})


#def transform_tica(input_directory, tica_directory, output_directory, lag, ndim):
#    """
#    Does a TICA projection of the data
#    
#    input_directory: directory with input data files
#    tica_directory: directory to store covariance and mean files
#    output_directory: directory to write transformed data files to
#    lag: TICA lagtime
#    ndim: number of TICA dimensions to use
#    """
#    tica = create_tica_transform(input_directory, tica_directory, output_directory, lag, ndim)
#    tica.update();
