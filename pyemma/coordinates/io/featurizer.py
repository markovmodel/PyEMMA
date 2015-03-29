from pyemma.util.annotators import deprecated
__author__ = 'Frank Noe, Martin Scherer'

import mdtraj
from mdtraj.geometry.dihedral import _get_indices_phi, \
    _get_indices_psi, compute_dihedrals

import numpy as np
import warnings

from pyemma.util.log import getLogger

log = getLogger('Featurizer')

__all__ = ['MDFeaturizer',
           'CustomFeature',
           ]


def _describe_atom(topology, index):
    """
    Returns a string describing the given atom

    :param topology:
    :param index:
    :return:
    """
    at = topology.atom(index)
    return "%s %i %s %i" % (at.residue.name, at.residue.index, at.name, at.index)


class CustomFeature(object):

    """
    A CustomFeature is the base class for all self defined features. You shall
    calculate your quantities in map method and return it as an ndarray.

    If you simply want to call a function with arguments, you do not need to derive.


    Parameters
    ----------
    func : function
        will be invoked with given args and kwargs on mapping traj
    args : list of positional args (optional)
    kwargs : named arguments

    Examples
    --------
    We use a FeatureReader to read md-trajectories and add a CustomFeature to
    transform coordinates:

    >>> reader = FeatureReader(...)
    >>> my_feature = CustomFeature(lambda x: 1.0 / x**2)
    >>> reader.featurizer.add_custom_feature(my_feature, output_dimension=3)

    Now a pipeline using this reader will apply the :math:`1 / x^2` transform on
    every frame being red.

    """

    def __init__(self, func=None, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def describe(self):
        return "override me to get proper description!"

    def map(self, traj):
        feature = self._func(traj, self._args, self._kwargs)
        assert isinstance(feature, np.ndarray)
        return feature


class SelectionFeature:
    """
    Just provide the cartesian coordinates of a selection of atoms (could be simply all atoms).
    The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

    """
    #TODO: Needs an orientation option

    def __init__(self, top, indexes):
        self.top = top
        self.indexes = np.array(indexes)
        self.prefix_label = "ATOM:"

    def describe(self):
        labels = []
        for i in self.indexes:
            labels.append("%s%s" % (self.prefix_label, _describe_atom(self.top, i)))
        return labels

    def map(self, traj):
        newshape = (traj.xyz.shape[0], 3*self.indexes.shape[0])
        return np.reshape(traj.xyz[:,self.indexes,:], newshape)


class DistanceFeature:

    def __init__(self, top, distance_indexes, periodic = True):
        self.top = top
        self.distance_indexes = np.array(distance_indexes)
        self.prefix_label = "DIST:"
        self.periodic = periodic

    def describe(self):
        labels = []
        for pair in self.distance_indexes:
            labels.append("%s %s - %s" % (self.prefix_label,
                                          _describe_atom(self.top, pair[0]),
                                          _describe_atom(self.top, pair[1])))
        return labels

    def map(self, traj):
        return mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)


class InverseDistanceFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, periodic = True):
        DistanceFeature.__init__(self, top, distance_indexes, periodic=periodic)
        self.prefix_label = "INVDIST:"

    def map(self, traj):
        return 1.0 / mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)


class ContactFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, threshold = 5.0, periodic=True):
        DistanceFeature.__init__(self, top, distance_indexes)
        self.prefix_label = "CONTACT:"
        self.threshold = threshold
        self.periodic = periodic

    def map(self, traj):
        dists = mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)
        res = np.zeros((len(traj), self.distance_indexes.shape[0]), dtype=np.float32)
        I = np.argwhere(dists <= self.threshold)
        res[I[:,0],I[:,1]] = 1.0
        return res


class AngleFeature:

    def __init__(self, top, angle_indexes):
        self.top = top
        self.angle_indexes = np.array(angle_indexes)

    def describe(self):
        labels = []
        for triple in self.angle_indexes:
            labels.append("ANGLE: %s - %s - %s " %
                          (_describe_atom(self.top, triple[0]),
                           _describe_atom(self.top, triple[1]),
                           _describe_atom(self.top, triple[2])))

        return labels

    def map(self, traj):
        return mdtraj.compute_angles(traj, self.angle_indexes)


class BackboneTorsionFeature:

    def __init__(self, topology):
        self.topology = topology

        # this is needed for get_indices functions, since they expect a Trajectory,
        # not a Topology
        class fake_traj():
            def __init__(self, top):
                self.top = top

        ft = fake_traj(topology)
        _, indices = _get_indices_phi(ft)
        self._phi_inds = indices

        _, indices = _get_indices_psi(ft)
        self._psi_inds = indices

        self.dim = len(self._phi_inds) + len(self._psi_inds)

    def describe(self):
        top = self.topology
        labels_phi = ["PHI %s %i" % (top.atom(ires[0]).residue.name, ires[0])
                      for ires in self._phi_inds]

        labels_psi = ["PHI %s %i" % (top.atom(ires[0]).residue.name, ires[0])
                      for ires in self._psi_inds]

        return labels_phi + labels_psi

    def map(self, traj):
        y1 = compute_dihedrals(traj, self._phi_inds).astype(np.float32)
        y2 = compute_dihedrals(traj, self._psi_inds).astype(np.float32)
        return np.hstack((y1, y2))


class MDFeaturizer(object):

    """extracts features from MD trajectories.

    Parameters
    ----------

    topfile : str
        a path to a topology file (pdb etc.)
    """

    def __init__(self, topfile):
        self.topology = (mdtraj.load(topfile)).topology

        #self.distance_indexes = []
        #self.inv_distance_indexes = []
        #self.contact_indexes = []
        #self.angle_indexes = []

        self.active_features = []
        self._dim = 0

    def describe(self):
        """
        Returns a list of strings, one for each feature selected,
        with human-readable descriptions of the features.

        Returns
        -------
        labels : list of str
            An ordered list of strings, one for each feature selected,
            with human-readable descriptions of the features.

        """
        labels = []

        for f in self.active_features:
            labels.append(f.describe())

        return labels

    def select(self, selstring):
        """
        Returns the indexes of atoms matching the given selection

        Parameters
        ----------
        selstring : str
            Selection string. See mdtraj documentation for details: http://mdtraj.org/latest/atom_selection.html

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select(selstring)

    def select_Ca(self):
        """
        Returns the indexes of all Ca-atoms

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("name CA")

    def select_Backbone(self):
        """
        Returns the indexes of backbone C, CA and N atoms

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("backbone name C CA N")

    def select_Heavy(self):
        """
        Returns the indexes of all heavy atoms (Mass >= 2)

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("mass >= 2")

    @staticmethod
    def pairs(sel):
        """
        Creates all pairs between indexes, except for 1 and 2-neighbors

        Parameters
        ----------
        sel : ndarray((n), dtype=int)
            array with selected atom indexes

        Return:
        -------
        sel : ndarray((m,2), dtype=int)
            m x 2 array with all pair indexes between different atoms that are at least 3 indexes apart,
            i.e. if i is the index of an atom, the pairs [i,i-2], [i,i-1], [i,i], [i,i+1], [i,i+2], will
            not be in sel. Moreover, the list is non-redundant, i.e. if [i,j] is in sel, then [j,i] is not.

        """
        p = []
        for i in range(len(sel)):
            for j in range(i+1,len(sel)):
                # get ordered pair
                I = sel[i]
                J = sel[j]
                if (I > J):
                    I = sel[j]
                    J = sel[i]
                # exclude 1 and 2 neighbors
                if (J > I+2):
                    p.append([I,J])
        return np.array(p)

    def add_all(self):
        """
        Adds all atom coordinates to the feature list.
        The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        """
        # TODO: add possibility to align to a reference structure
        self.add_selection(range(self.topology.n_atoms))

    def add_selection(self, indexes):
        """
        Adds the selected atom coordinates to the feature list.
        The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Parameters
        ----------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        # TODO: add possibility to align to a reference structure
        f = SelectionFeature(self.topology, indexes)
        self.active_features.append(f)
        self._dim += np.shape(indexes)[0]*3

    @deprecated
    def distances(self, atom_pairs):
        return self.add_distances(atom_pairs)

    def add_distances(self, atom_pairs, periodic=True):
        """
        Adds the distances between the given pairs of atoms to the feature list.

        atom_pairs : ndarray((n,2), dtype=int)
            n x 2 array with pairs of atoms between which the distances shall be computed

        """
        #assert atom_pairs.shape ==...
        f = DistanceFeature(self.topology, atom_pairs, periodic=periodic)
        self.active_features.append(f)
        self._dim += np.shape(atom_pairs)[0]

    @deprecated
    def distancesCa(self):
        return self.add_distances_ca()

    def add_distances_ca(self, periodic=True):
        """
        Adds the distances between all Ca's (except for 1- and 2-neighbors) to the feature list.

        """
        distance_indexes = self.pairs(self.select_Ca())
        self.add_distances(distance_indexes, periodic=periodic)

    @deprecated
    def inverse_distances(self, atom_pairs):
        return self.add_inverse_distances(atom_pairs)

    def add_inverse_distances(self, atom_pairs, periodic=True):
        """
        Adds the inverse distances between the given pairs of atoms to the feature list.

        atom_pairs : ndarray((n,2), dtype=int)
            n x 2 array with pairs of atoms between which the inverse distances shall be computed

        """
        f = InverseDistanceFeature(self.topology, atom_pairs, periodic=True)
        self.active_features.append(f)
        self._dim += np.shape(atom_pairs)[0]

    @deprecated
    def contacts(self, atom_pairs):
        return self.add_contacts(atom_pairs)

    def add_contacts(self, atom_pairs, threshold = 5.0, periodic=True):
        """
        Adds the set of contacts to the feature list

        Parameters
        ----------
        atom_pairs : ndarray((n, 2), dtype=int)
            n x 2 array of pairs of atoms to compute contacts between
        threshold : float, optional, default = 5.0
            distances below this threshold will result in a feature 1.0, distances above will result in 0.0.
            The default is set with Angstrom distances in mind.
            Make sure that you know whether your coordinates are in Angstroms or nanometers when setting this threshold.

        """
        #assert atom_pairs.shape == ...
        #assert in_bounds , ... 
        f = ContactFeature(self.topology, atom_pairs, threshold=threshold, periodic=periodic)
        self.active_features.append(f)
        self._dim += np.shape(atom_pairs)[0]

    @deprecated
    def angles(self, indexes):
        return self.add_angles(indexes)

    def add_angles(self, indexes):
        """
        Adds the list of angles to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 3), dtype=int
            an array with triplets of atom indices

        """
        #assert indexes.shape == 

        f = AngleFeature(self.topology, indexes)
        self.active_features.append(f)
        self._dim += np.shape(indexes)[0]

    @deprecated
    def backbone_torsions(self):
        return self.add_backbone_torsions()

    def add_backbone_torsions(self):
        """
        Adds all backbone phi/psi angles to the feature list.

        """
        f = BackboneTorsionFeature(self.topology)
        self.active_features.append(f)
        self._dim += f.dim

    def add_custom_feature(self, feature, output_dimension):
        """
        Adds a custom feature to the feature list.

        Parameters
        ----------
        feature : object
            an object with interface like CustomFeature (map, describe methods)
        output_dimension : int
            a mapped feature coming from has this dimension.

        """
        assert output_dimension > 0, "tried to add empty feature"
        assert hasattr(feature, 'map'), "no map method in given feature"
        assert hasattr(feature, 'describe')

        self.active_features.append(feature)
        self._dim += output_dimension

    def dimension(self):
        """ current dimension due to selected features

        Returns
        -------
        dim : int
            total dimension due to all selection features

        """
        return self._dim

    def map(self, traj):
        """
        Maps an mdtraj Trajectory object to the selected output features

        Parameters
        ----------
        traj : mdtraj Trajectory
            Trajectory object used as an input

        Returns
        -------
        out : ndarray( ( T, n ), dtype = float32 )
            Output features: For each of T time steps in the given trajectory, a vector with all n output features
            selected.

        """
        # if there are no features selected, return given trajectory
        if self._dim == 0:
            warnings.warn("You have no features selected. Returning plain coordinates.")
            return traj.xyz

        # TODO: define preprocessing step (RMSD etc.)

        # otherwise build feature vector.
        feature_vec = []

        # only iterate over unique
        # TODO: implement __hash__ and __eq__, see
        # https://stackoverflow.com/questions/2038010/sets-of-instances
        for f in set(self.active_features):
            feature_vec.append(f.map(traj).astype(np.float32))

        return np.hstack(feature_vec)
