'''
Created on 20.11.2013

@author: marscher
'''

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

"""
# some additional  algorithm specific parameters
                + "[ -algorithm kmeans -clustercenters <int> [-metric {euclidean}] [-maxiterations <int>]\n"
                + " |-algorithm kcenters -clustercenters <int> -metric [ minrmsd | euclidean ]\n"
                + " |-algorithm regularspatial -dmin <double> -metric [ minrmsd | euclidean ]\n"
                + " |-algorithm regulartemporal -spacing <int> -metric [ minrmsd | euclidean ]\n"
"""
algorithms = ['kcenter', 'kmeans', 'regularspatial', 'regulartemporal']

memory_sensitive = ['kcenter', 'kmeans']
input_formats = ['xtc', 'dcd' , 'ascii' , 'ensembleascii', 'auto']
ascii_formats = ['ascii' , 'ensembleascii']

metrics = ['rmsd', 'euclidian']

v_inputtraj = []
v_inputformat = ''
v_clustercenters = 0
v_dmin = 0.0
v_spacing = 0
v_max_interations = 1000
v_outputfolder= ''
v_stepwidthForClustering = 0


if __name__ == '__main__':
    #does not work
    #print algorithms[memory_sensitive]
    pass