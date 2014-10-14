'''
Created on 04.12.2013

@author: Susanna Roeblitz, Marcus Weber

taken from ZIBMolPy which can also be found on Github:
https://github.com/CMD-at-ZIB/ZIBMolPy/blob/master/ZIBMolPy_package/ZIBMolPy/algorithms.py
'''

import numpy as np

#===============================================================================
def cluster_by_isa(eigenvectors, n_clusters):
    #TODO: check this somehow, probably more args nessecary
    # eigenvectors have to be sorted in descending order in regard to their eigenvalues
    if n_clusters > len(eigenvectors):
        n_clusters = len(eigenvectors)

    # the actual ISA algorithm
    c = eigenvectors[:, range(n_clusters)]
    ortho_sys = np.copy(c)
    max_dist = 0.0
    ind = np.zeros(n_clusters, dtype=np.int32)

    # first two representatives with maximum distance
    for (i, row) in enumerate(c):
        if np.linalg.norm(row, 2) > max_dist:
            max_dist = np.linalg.norm(row, 2)
            ind[0] = i

    ortho_sys -= c[ind[0], None]

    # further representatives via Gram-Schmidt orthogonalization
    for k in range(1, n_clusters):
        max_dist = 0.0
        temp = np.copy(ortho_sys[ind[k-1]])

        for (i, row) in enumerate(ortho_sys):
            row -= np.dot( np.dot(temp, np.transpose(row)), temp )
            distt = np.linalg.norm(row, 2)
            if distt > max_dist and i not in ind[0:k]:
                max_dist = distt
                ind[k] = i

        ortho_sys /= np.linalg.norm( ortho_sys[ind[k]], 2 )

    # linear transformation of eigenvectors
    rot_mat = np.linalg.inv(c[ind])

    chi = np.dot(c, rot_mat)

    # determining the indicator
    indic = np.min(chi)
    # Defuzzifizierung der Zugehoerigkeitsfunktionen
    #[minVal cF]=max(transpose(Chi)); #TODO minval? Marcus-check
    #minVal = np.max(np.transpose(chi))
    c_f = np.amax(np.transpose(chi))

    return (c_f, indic, chi, rot_mat)


#===============================================================================
def opt_soft(eigvectors, rot_matrix, n_clusters):

    # only consider first n_clusters eigenvectors
    eigvectors = eigvectors[:,:n_clusters]

    # crop first row and first column from rot_matrix
    rot_crop_matrix = rot_matrix[1:,1:]

    (x, y) = rot_crop_matrix.shape

    # reshape rot_crop_matrix into linear vector
    rot_crop_vec = np.reshape(rot_crop_matrix, x*y)

    # target function for optimization
    def susanna_func(rot_crop_vec, eigvectors):
        # reshape into matrix
        rot_crop_matrix = np.reshape(rot_crop_vec, (x, y))
        # fill matrix
        rot_matrix = fill_matrix(rot_crop_matrix, eigvectors)

        result = 0
        for i in range(0, n_clusters):
            for j in range(1, n_clusters):
                result += np.power(rot_matrix[j,i], 2) / rot_matrix[0,i]
        return -result


    from scipy.optimize import fmin
    rot_crop_vec_opt = fmin( susanna_func, rot_crop_vec, args=(eigvectors,), disp=False)

    rot_crop_matrix = np.reshape(rot_crop_vec_opt, (x, y))
    rot_matrix = fill_matrix(rot_crop_matrix, eigvectors)

    return rot_matrix


#===============================================================================
def fill_matrix(rot_crop_matrix, eigvectors):

    (x, y) = rot_crop_matrix.shape

    row_sums = np.sum(rot_crop_matrix, axis=1)
    row_sums = np.reshape(row_sums, (x,1))

    # add -row_sums as leftmost column to rot_crop_matrix
    rot_crop_matrix = np.concatenate((-row_sums, rot_crop_matrix), axis=1 )

    tmp = -np.dot(eigvectors[:,1:], rot_crop_matrix)

    tmp_col_max = np.max(tmp, axis=0)
    tmp_col_max = np.reshape(tmp_col_max, (1,y+1))

    tmp_col_max_sum = np.sum(tmp_col_max)

    # add col_max as top row to rot_crop_matrix and normalize
    rot_matrix = np.concatenate((tmp_col_max, rot_crop_matrix), axis=0 )
    rot_matrix /= tmp_col_max_sum

    return rot_matrix

