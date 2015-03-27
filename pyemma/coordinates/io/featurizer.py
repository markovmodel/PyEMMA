__author__ = 'noe, marscher'

import mdtraj
from mdtraj.geometry.dihedral import _get_indices_phi, \
    _get_indices_psi, compute_dihedrals

import numpy as np
import warnings

from pyemma.util.annotators import deprecated
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


def _catch_unhashable(x):
    if hasattr(x, '__getitem__'):
        res = list(x)
        for i, value in enumerate(x):
            if isinstance(value, np.ndarray):
                res[i] = _hash_numpy_array(value)
            else:
                res[i] = value
        return tuple(res)
    elif isinstance(x, np.ndarray):
        return _hash_numpy_array(value)

    return x


def _hash_numpy_array(x):
    x.flags.writeable = False
    hash_value = hash(x.shape)
    hash_value |= hash(x.strides)
    hash_value |= hash(x.data)
    return hash_value


class CustomFeature(object):

    """
    A CustomFeature is the base class for all self defined features. You shall
    calculate your quantities in map method and return it as an ndarray.

    If you simply want to call a function with arguments, you do not need to derive.


    Parameters
    ----------
    func : function
        will be invoked with given args and kwargs on mapping traj
    args : list of positional args (optional) passed to func
    kwargs : named arguments (optional) passed to func

    Examples
    --------
    We use a FeatureReader to read MD-trajectories and add a CustomFeature to
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
        return ["override me to get proper description!"]

    def map(self, traj):
        feature = self._func(traj, self._args, self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("your function should return a NumPy array!")
        return feature

    def __hash__(self):
        hash_value = hash(self._func)
        # if key contains numpy arrays, we hash their data arrays
        key = tuple(map(_catch_unhashable, self._args) +
                    map(_catch_unhashable, sorted(self._kwargs.items())))
        hash_value |= hash(key)
        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class DistanceFeature(object):

    def __init__(self, top, distance_indexes):
        self.top = top
        self.distance_indexes = np.array(distance_indexes)
        self.prefix_label = "DIST:"

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  _describe_atom(self.top, pair[0]),
                                  _describe_atom(self.top, pair[1]))
                  for pair in self.distance_indexes]
        return labels

    def map(self, traj):
        return mdtraj.compute_distances(traj, self.distance_indexes)

    def __hash__(self):
        hash_value = _hash_numpy_array(self.distance_indexes)
        hash_value |= hash(self.top)
        hash_value |= hash(self.prefix_label)
        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InverseDistanceFeature(DistanceFeature):

    def __init__(self, top, distance_indexes):
        DistanceFeature.__init__(self, top, distance_indexes)
        self.prefix_label = "INVDIST:"

    def map(self, traj):
        return 1.0 / mdtraj.compute_distances(traj, self.distance_indexes)

    # does not need own hash impl, since we take prefix label into account


class BackboneTorsionFeature(object):

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
        labels_phi = ["PHI %s %i" % _describe_atom(top, ires)
                      for ires in self._phi_inds]

        labels_psi = ["PSI %s %i" % _describe_atom(top, ires)
                      for ires in self._psi_inds]

        return labels_phi + labels_psi

    def map(self, traj):
        y1 = compute_dihedrals(traj, self._phi_inds).astype(np.float32)
        y2 = compute_dihedrals(traj, self._psi_inds).astype(np.float32)
        return np.hstack((y1, y2))

    def __hash__(self):
        hash_value = _hash_numpy_array(self._phi_inds)
        hash_value |= _hash_numpy_array(self._psi_inds)
        hash_value |= hash(self.topology)

        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class MDFeaturizer(object):

    """extracts features from MD trajectories.

    Parameters
    ----------

    topfile : str
        a path to a topology file (pdb etc.)
    """

    def __init__(self, topfile):
        self.topology = (mdtraj.load(topfile)).topology
        self.active_features = set()
        self._dim = 0

    def describe(self):
        """
        Returns a list of strings, one for each feature selected,
        with human-readable descriptions of the features.

        Returns
        -------
        labels : list
            a list of labels of all active features
        """
        labels = [f.describe() for f in self.active_features]
        return labels

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
        # FIXME: this string raises
        return self.topology.select("backbone name C[A] N")

    def select_Heavy(self):
        """
        Selects all heavy atoms

        :return:
        """
        return self.topology.select("mass >= 2")

    @staticmethod
    def pairs(sel):
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

        return pairs

    def _check_indices(self, pair_inds, pair_n=2):
        """ensure pairs are valid (shapes, all atom indices available?, etc.) 
        """
        pair_inds = np.array(pair_inds)

        if pair_inds.ndim != 2:
            raise ValueError("pair indices has to be a matrix.")

        if pair_inds.shape[1] != pair_n:
            raise ValueError("pair indices shape has to be (x, %i)." % pair_n)

        if pair_inds.max() > self.topology.n_atoms:
            raise ValueError("index out of bounds: %i."
                             " Maximum atom index available: %i"
                             % (pair_inds.max(), self.topology.n_atoms))

        return pair_inds

    @deprecated
    def distances(self, atom_pairs):
        return self.add_distances(atom_pairs)

    def add_distances(self, atom_pairs):
        """
        Adds the set of distances to the feature list

        Parameters
        ----------
        atom_pairs : ndarray (n, 2)
            pair of indices, n has to be smaller than n atoms of topology.
        """
        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.topology, distance_indexes=atom_pairs)

        if f not in self.active_features:
            self.active_features.add(f)
            self._dim += np.shape(atom_pairs)[0]
        else:
            log.warning("tried to add duplicate feature")

    @deprecated
    def distancesCa(self):
        return self.add_distances_ca()

    def add_distances_ca(self):
        """
        Adds the set of Ca-distances to the feature list
        """
        pairs = self.pairs(self.select_Ca())

        f = DistanceFeature(self.topology, distance_indexes=pairs)

        if f not in self.active_features:
            self.active_features.add(f)
            self._dim += np.shape(self.pairs)[0]
        else:
            log.warning("tried to add duplicate feature")

    @deprecated
    def inverse_distances(self, atom_pairs):
        return self.add_inverse_distances(atom_pairs)

    def add_inverse_distances(self, atom_pairs):
        r"""
        Adds the set of inverse distances to the feature list

        Parameters
        ----------
        atom_pairs : ndarray (n, 2)
            pair of indices, n has to be smaller than n atoms of topology.
        """
        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(top=self.topology,
                                   distance_indexes=atom_pairs)

        if f not in self.active_features:
            self.active_features.add(f)
            self._dim += np.shape(atom_pairs)[0]
        else:
            log.warning("tried to add duplicate feature")

    @deprecated
    def contacts(self, atom_pairs):
        return self.add_contacts(atom_pairs)

    def add_contacts(self, atom_pairs):
        r"""
        Adds the set of contacts to the feature list.

        Parameters
        ----------
        atom_pairs : ndarray (n, 2)
            pair of indices, n has to be smaller than n atoms of topology.
        """
        atom_pairs = self._check_indices(atom_pairs)
        f = CustomFeature(mdtraj.compute_contacts, atom_pairs=atom_pairs)

        def describe():
            labels = ["CONTACT: %s - %s" %
                      (_describe_atom(self.topology, pair[0]),
                       _describe_atom(self.topology, pair[1]))
                      for pair in atom_pairs]
            return labels
        f.describe = describe
        f.topology = self.topology

        if f not in self.active_features:
            self._dim += np.shape(atom_pairs)[0]
            self.active_features.add(f)
        else:
            log.warning("tried to add duplicate feature")

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
        indexes = self._check_indices(indexes, pair_n=3)
        f = CustomFeature(mdtraj.compute_angles, indexes=indexes)

        def describe():
            labels = ["ANGLE: %s - %s - %s " %
                      (_describe_atom(self.topology, triple[0]),
                       _describe_atom(self.topology, triple[1]),
                       _describe_atom(self.topology, triple[2]))
                      for triple in indexes]

            return labels

        f.describe = describe
        f.topology = self.topology

        if f not in self.active_features:
            self.active_features.add(f)
            self._dim += np.shape(indexes)[0]
        else:
            log.warning("tried to add duplicate feature")

    @deprecated
    def backbone_torsions(self):
        return self.add_backbone_torsions()

    def add_backbone_torsions(self):
        """
        Adds all backbone torsions
        """
        f = BackboneTorsionFeature(self.topology)
        if f not in self.active_features:
            self.active_features.add(f)
            self._dim += f.dim
        else:
            log.warning("tried to add duplicate feature")

    def add_custom_feature(self, feature, output_dimension):
        """
        Parameters
        ----------
        feature : object
            an object with interface like CustomFeature (map, describe methods)
        output_dimension : int
            a mapped feature coming from has this dimension.
        """
        if output_dimension <= 0:
            raise ValueError("output_dimension has to be positive")

        if not hasattr(feature, 'map'):
            raise ValueError("no map method in given feature")
        else:
            if not callable(getattr(feature, 'map')):
                raise ValueError("map exists but is not a method")

        if feature not in self.active_features:
            self.active_features.add(feature)
            self._dim += output_dimension
        else:
            log.warning("tried to add duplicate feature")

    def dimension(self):
        """ current dimension due to selected features """
        return self._dim

    def map(self, traj):
        """
        Computes the features for the given trajectory
        :return:
        """
        # if there are no features selected, return given trajectory
        if self._dim == 0:
            warnings.warn(
                "You have no features selected. Returning plain coordinates.")
            return traj.xyz

        # TODO: define preprocessing step (RMSD etc.)

        # otherwise build feature vector.
        feature_vec = []

        # TODO: consider parallel evaluation computation here, this effort is
        # only worth it, if computation time dominates memory transfers
        for f in self.active_features:
            feature_vec.append(f.map(traj).astype(np.float32))

        return np.hstack(feature_vec)
