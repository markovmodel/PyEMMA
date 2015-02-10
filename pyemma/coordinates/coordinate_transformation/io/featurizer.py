__author__ = 'noe'

import mdtraj
import numpy as np

__all__ = ['Featurizer']


class MDFeaturizer(object):

    """
    TODO: This class is currently not easily extensible, because all features need to be explicitly implemented and
    changes need to be made in both describe() and map(). It would be better to have a feature list that can be added
    to. Moreover, it would be good to have own feature-implementations that can be added.
    
    MS: a feature might consists out of a tuple (callback function -> eg. mdtraj feature calculation and arguments).
    These are maintained in a list, so we easily iterate over this list in map()

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


    def _describe_atom(self, index):
        """
        Returns a string describing the given atom

        :param index:
        :return:
        """
        at = self.topology.atom(index)
        return at.residue.name+" "+str(at.residue.index)+" "+at.name+" "+str(at.index)


    def describe(self):
        """
        Returns a list of strings, one for each feature selected, with human-readable descriptions of the features.

        :return:
        """
        labels = []

        # distances
        if (self.use_distances):
            for pair in self.distance_indexes:
                labels.append("DIST: "+self._describe_atom(pair[0])+" - "+self._describe_atom(pair[1]))

        # inverse distances
        if (self.use_inv_distances):
            for pair in self.inv_distance_indexes:
                labels.append("INVDIST: "+self._describe_atom(pair[0])+" - "+self._describe_atom(pair[1]))

        # contacts
        if (self.use_contacts):
            for pair in self.contact_indexes:
                labels.append("CONTACT: "+self._describe_atom(pair[0])+" - "+self._describe_atom(pair[1]))

        # angles
        if (self.use_angles):
            for triple in self.angle_indexes:
                labels.append("ANGLE: "+self._describe_atom(triple[0])
                              +" - "+self._describe_atom(triple[1])
                              +" - "+self._describe_atom(triple[2]))

        # backbone torsions
        if (self.use_backbone_torsions):
            phis = self._list_phis()
            for ires in phis:
                labels.append("PHI: "+self.topology.residue(ires).name+" "+str(ires))
            psis = self._list_psis()
            for ires in psis:
                labels.append("PSI: "+self.topology.residue(ires).name+" "+str(ires))


    def select(self, selstring):
        """
        Selects using the given selection string

        :param selstring:
        :return:
        """
        return self.topology.select(selstring)


    def select_Ca(self):
        """
        Selects Ca atoms

        :return:
        """
        return self.topology.select("name CA")


    def select_Backbone(self):
        """
        Selects backbone C, CA and N atoms

        :return:
        """
        return self.topology.select("name C CA N")


    def select_Heavy(self):
        """
        Selects all heavy atoms

        :return:
        """
        return self.topology.select("mass >= 2")


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


    def _has_atom(self, res_index, name):
        for atom in self.topology.atoms():
            if atom.name.lower() == name.lower():
                return True
        return False


    def _list_phis(self):
        # phi: C-1, N, CA, C
        phis = []
        for ires in range(1, self.topology.n_residues):
            if (self._has_atom(ires-1, "C") and self._has_atom(ires, "N") and
                self._has_atom(ires, "CA") and self._has_atom(ires, "C")):
                phis.append(ires)
        return phis


    def _list_psis(self):
        # psi: N, CA, C, N+1
        psis = []
        for ires in range(0, self.topology.n_residues-1):
            if (self._has_atom(ires, "N") and self._has_atom(ires, "CA") and
                self._has_atom(ires, "C") and self._has_atom(ires+1, "N")):
                psis.append(ires)
        return psis


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


    def distancesCa(self):
        """
        Adds the set of Ca-distances to the feature list

        :param atom_pairs:
        :return:
        """
        self.use_distances = True
        self.distance_indexes = self.pairs(self.select_Ca())
        self.dim += np.shape(self.distance_indexes)[0]


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
        self.dim += 2 * self.topology.n_residues


    def dimension(self):
        return self.dim


    def map(self, traj):
        """
        Computes the features for the given trajectory
        TODO: why enforce single precision? - FN: Because this is the most memory-consuming part of the pipeline and single precision is way (!) more than enough. In fact np.float16 might be enough
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
