import numpy as np

import plotting

import matplotlib.pyplot as plt

if __name__=="__main__":
    centers=np.loadtxt('grid_centers20x20.dat')
    lcc=np.loadtxt('lcc.dat').astype(int)
    committor=np.load('forward_committor_c5_p2.npy')

    centers=centers[lcc, :]

    plotting.free_energy(centers, committor, levels=np.linspace(0.0, 1.0, 11), fill_value=0.0, method='cubic')
    plt.show()
