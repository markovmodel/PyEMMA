__author__ = 'noe'

import mdtraj
import numpy as np


class Featurizer(object):

    """

    """

    def __init__(self, topfile):
        """
        Parameters
        ----------

        topfile : str
            a topology file (pdb etc.)

        """
        self.topology = (mdtraj.load(topfile)).topology

        self.use_distances = False
        self.distance_indexes = []

        self.use_inv_distances = False
        self.inv_distance_indexes = []

        self.use_contacts = False
        self.contact_indexes = []

        self.use_angles = False
        self.angle_indexes = []

        self.use_backbone_torsions = False

        self.dim = 0

    def describe(self):
        return "Featurizer[distances = ", self.use_distances, " contacts = ", \
            self.use_contacts, " angles = ", self.use_angles, \
            " backbone torsions = ", self.use_backbone_torsions

    def selCa(self):
        return self.topology.select("name CA")

    def selHeavy(self):
        return self.topology.select("mass >= 2")


    def distances(self, atom_pairs):
        """
        Adds the set of distances to the feature list

        :param atom_pairs:
        :return:
        """
        assert len(atom_pairs) > 0
        self.use_distances = True
        self.distance_indexes = atom_pairs
        self.dim += np.shape(atom_pairs)[0]
        assert self.dim > 0


    def inverse_distances(self, atom_pairs):
        """
        Adds the set of inverse distances to the feature list

        :param atom_pairs:
        :return:
        """
        assert len(atom_pairs) > 0
        self.use_inv_distances = True
        self.inv_distance_indexes = atom_pairs
        self.dim += np.shape(atom_pairs)[0]
        assert self.dim > 0


    def pairs(self, sel):
        """
        Creates all pairs between indexes, except for 1 and 2-neighbors

        :return:
        """
        nsel = np.shape(sel)[0]
        npair = ((nsel - 2) * (nsel - 3)) / 2
        pairs = np.zeros((npair, 2))
        s = 0
        for i in range(0, nsel - 3):
            d = nsel - 3 - i
            pairs[s:s + d, 0] = sel[i]
            pairs[s:s + d, 1] = sel[i + 3:nsel]
            s += d

        assert len(pairs) > 0
        return pairs

    def contacts(self, atom_pairs):
        """
        Adds the set of contacts to the feature list
        :param atom_pairs:
        :return:
        """
        self.use_contacts = True
        self.contact_indexes = atom_pairs
        self.dim += np.shape(atom_pairs)[0]

    def angles(self, indexes):
        """
        Adds the list of angles to the feature list

        :param indexes:
        :return:
        """
        self.use_angles = True
        self.angle_indexes = indexes
        self.dim += np.shape(indexes)[0]

    def backbone_torsions(self):
        """
        Adds all backbone torsions

        :return:
        """
        self.use_backbone_torsions = True
        self.dim += 2 * mdtraj.n_residues

    def dimension(self):
        return self.dim

    def map(self, traj):
        """
        Computes the features for the given trajectory
        TODO: why enforce single precision?
        :return:
        """
        Y = None

        # distances
        if (self.use_distances):
            y = mdtraj.compute_distances(traj, self.distance_indexes)
            Y = y.astype(np.float32)

        # inverse distances
        if (self.use_inv_distances):
            y = 1.0 / mdtraj.compute_distances(traj, self.inv_distance_indexes)
            Y = y.astype(np.float32)

        # contacts
        if (self.use_contacts):
            y = mdtraj.compute_contacts(traj, self.contact_indexes)
            y = y.astype(np.float32)
            if Y is None:
                Y = y
            else:
                Y = np.concatenate((Y, y), axis=1)

        # angles
        if (self.use_angles):
            y = mdtraj.compute_angles(traj, self.angle_indexes)
            y = y.astype(np.float32)
            if Y is None:
                Y = y
            else:
                Y = np.concatenate((Y, y), axis=1)

        # backbone torsions
        if (self.use_backbone_torsions):
            y1 = mdtraj.compute_phi(traj)
            y1 = y1.astype(np.float32)
            y2 = mdtraj.compute_psi(traj)
            y2 = y2.astype(np.float32)
            if Y is None:
                Y = y1
            else:
                Y = np.concatenate((Y, y1), axis=1)
            Y = np.concatenate((Y, y2), axis=1)

        return Y
