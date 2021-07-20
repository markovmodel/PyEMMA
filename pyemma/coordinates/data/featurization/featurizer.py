# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import warnings

from pyemma._base.loggable import Loggable
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.util.annotators import deprecated
from pyemma.util.types import is_string
import mdtraj

from pyemma.coordinates.data.featurization.util import (_parse_pairwise_input,
                                                        _parse_groupwise_input)

from .misc import CustomFeature
import numpy as np
from pyemma.coordinates.util.patches import load_topology_cached
from mdtraj import load_topology as load_topology_uncached


__author__ = 'Frank Noe, Martin Scherer'
__all__ = ['MDFeaturizer']


class MDFeaturizer(SerializableMixIn, Loggable):
    r"""Extracts features from MD trajectories."""
    __serialize_version = 0
    __serialize_fields = ('use_topology_cache',
                         '_topologyfile',
                         'topology',
                         'active_features',
                          )

    def __init__(self, topfile, **kwargs):
        """extracts features from MD trajectories.

        Parameters
        ----------

        topfile : str or mdtraj.Topology
           a path to a topology file (pdb etc.) or an mdtraj Topology() object
        use_cache : boolean, default=True
           Deprecated, topologies are always cached.
        """
        self.topology = None
        self.topologyfile = topfile
        self.active_features = []

    @property
    def topologyfile(self):
        return self._topologyfile

    @topologyfile.setter
    def topologyfile(self, topfile):
        self._topologyfile = topfile
        if isinstance(topfile, str):
            self.topology = load_topology_cached(topfile)
            self._topologyfile = topfile
        elif isinstance(topfile, mdtraj.Topology):
            self.topology = topfile
        elif isinstance(topfile, mdtraj.Trajectory):
            self.topology = topfile.topology
        else:
            raise ValueError("no valid topfile arg: type was %s, "
                             "but only string or mdtraj.Topology allowed." % type(topfile))

    def __add_feature(self, f):
        # perform sanity checks
        if f.dimension == 0:
            self.logger.error("given an empty feature (eg. due to an empty/"
                               "ineffective selection). Skipping it."
                               " Feature desc: %s" % f.describe())
            return

        if f not in self.active_features:
            self.active_features.append(f)
        else:
            self.logger.warning("tried to re-add the same feature %s"
                                % f.__class__.__name__)

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
        all_labels = []
        for f in self.active_features:
            all_labels += f.describe()
        return all_labels

    def __repr__(self):
        import pprint
        feat_str = pprint.pformat(self.describe()[:10])[:-1] + ', ...]'
        return 'MDFeaturizer with features:\n{feats}'.format(feats=feat_str)

    def select(self, selstring):
        """
        Returns the indexes of atoms matching the given selection

        Parameters
        ----------
        selstring : str
            Selection string. See mdtraj documentation for details:
            http://mdtraj.org/latest/atom_selection.html

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
        return self.topology.select("backbone and (name C or name CA or name N)")

    def select_Heavy(self, exclude_symmetry_related=False):
        """
        Returns the indexes of all heavy atoms (Mass >= 2),
        optionally excluding symmetry-related heavy atoms.

        Parameters
        ----------
        exclude_symmetry_related : boolean, default=False
            if True, exclude symmetry-related heavy atoms.

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        if exclude_symmetry_related:
            exclusions = []

            exclusions.append("mass < 2")
            exclusions.append("(resname == VAL and name == CG)")
            exclusions.append("(resname == LEU and name == CD)")
            exclusions.append("(resname == PHE and name == CD) or (resname == PHE and name == CE)")
            exclusions.append("(resname == TYR and name == CD) or (resname == TYR and name == CE)")
            exclusions.append("(resname == GLU and name == OD1) or (resname == GLU and name == OD2)")
            exclusions.append("(resname == ASP and name == OG1) or (resname == ASP and name == OG2)")
            exclusions.append("(resname == HIS and name == ND1) or (resname == HIS and name == NE2)")
            exclusions.append("(resname == ARG and name == NH1) or (resname == ARG and name == NH2)")

            exclusion_string = ' or '.join(exclusions)
            selection_string = 'not (' + exclusion_string + ')'

            return self.topology.select(selection_string)
        else:
            return self.topology.select("mass >= 2")

    @staticmethod
    def pairs(sel, excluded_neighbors=0):
        """
        Creates all pairs between indexes. Will exclude closest neighbors up to :py:obj:`excluded_neighbors`
        The self-pair (i,i) is always excluded

        Parameters
        ----------
        sel : ndarray((n), dtype=int)
            array with selected atom indexes

        excluded_neighbors: int, default = 0
            number of neighbors that will be excluded when creating the pairs

        Returns
        -------
        sel : ndarray((m,2), dtype=int)
            m x 2 array with all pair indexes between different atoms that are at least :obj:`excluded_neighbors`
            indexes apart, i.e. if i is the index of an atom, the pairs [i,i-2], [i,i-1], [i,i], [i,i+1], [i,i+2], will
            not be in :py:obj:`sel` (n=excluded_neighbors) if :py:obj:`excluded_neighbors` = 2.
            Moreover, the list is non-redundant,i.e. if [i,j] is in sel, then [j,i] is not.

        """

        assert isinstance(excluded_neighbors,int)

        p = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                # get ordered pair
                I = sel[i]
                J = sel[j]
                if (I > J):
                    I = sel[j]
                    J = sel[i]
                # exclude 1 and 2 neighbors
                if (J > I + excluded_neighbors):
                    p.append([I, J])
        return np.array(p)

    def _check_indices(self, pair_inds, pair_n=2):
        """ensure pairs are valid (shapes, all atom indices available?, etc.)
        """

        pair_inds = np.array(pair_inds).astype(dtype=int, casting='safe')

        if pair_inds.ndim != 2:
            raise ValueError("pair indices has to be a matrix.")

        if pair_inds.shape[1] != pair_n:
            raise ValueError("pair indices shape has to be (x, %i)." % pair_n)

        if pair_inds.max() > self.topology.n_atoms:
            raise ValueError("index out of bounds: %i."
                             " Maximum atom index available: %i"
                             % (pair_inds.max(), self.topology.n_atoms))

        return pair_inds

    def add_all(self, reference=None, atom_indices=None, ref_atom_indices=None):
        """
        Adds all atom coordinates to the feature list.
        The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Parameters
        ----------
        reference: mdtraj.Trajectory or None, default=None
            if given, all data is being aligned to the given reference with Trajectory.superpose
        atom_indices : array_like, or None
            The indices of the atoms to superpose. If not
            supplied, all atoms will be used.
        ref_atom_indices : array_like, or None
            Use these atoms on the reference structure. If not supplied,
            the same atom indices will be used for this trajectory and the
            reference one.
        """
        self.add_selection(list(range(self.topology.n_atoms)), reference=reference,
                           atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)

    def add_selection(self, indexes, reference=None, atom_indices=None, ref_atom_indices=None):
        """
        Adds the coordinates of the selected atom indexes to the feature list.
        The coordinates of the selection [1, 2, ...] are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Parameters
        ----------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes
        reference: mdtraj.Trajectory or None, default=None
            if given, all data is being aligned to the given reference with Trajectory.superpose
        atom_indices : array_like, or None
            The indices of the atoms to superpose. If not
            supplied, all atoms will be used.
        ref_atom_indices : array_like, or None
            Use these atoms on the reference structure. If not supplied,
            the same atom indices will be used for this trajectory and the
            reference one.
        """
        from .misc import SelectionFeature, AlignFeature
        if reference is None:
            f = SelectionFeature(self.topology, indexes)
        else:
            if not isinstance(reference, mdtraj.Trajectory):
                raise ValueError('reference is not a mdtraj.Trajectory object, but {}'.format(reference))
            f = AlignFeature(reference=reference, indexes=indexes,
                             atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)
        self.__add_feature(f)

    def add_distances(self, indices, periodic=True, indices2=None):
        r"""
        Adds the distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the distances shall be computed.

        periodic : optional, boolean, default is True
            If periodic is True and the trajectory contains unitcell information,
            distances will be computed under the minimum image convention.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the distances between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.


        .. note::
            When using the iterable of integers input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.
        """
        from .distances import DistanceFeature

        atom_pairs = _parse_pairwise_input(
            indices, indices2, self.logger, fname='add_distances()')

        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.topology, atom_pairs, periodic=periodic)
        self.__add_feature(f)

    def add_distances_ca(self, periodic=True, excluded_neighbors=2):
        """
        Adds the distances between all Ca's to the feature list.

        Parameters
        ----------
        periodic : boolean, default is True
            Use the minimum image convetion when computing distances

        excluded_neighbors : int, default is 2
            Number of exclusions when compiling the list of pairs. Two CA-atoms are considered
            neighbors if they belong to adjacent residues.

        """

        # Atom indices for CAs
        at_idxs_ca = self.select_Ca()
        # Residue indices for residues contatinig CAs
        res_idxs_ca = [self.topology.atom(ca).residue.index for ca in at_idxs_ca]
        # Pairs of those residues, with possibility to exclude neighbors
        res_idxs_ca_pairs = self.pairs(res_idxs_ca, excluded_neighbors=excluded_neighbors)
        # Mapping back pairs of residue indices to pairs of CA indices
        distance_indexes = []
        for ri, rj in res_idxs_ca_pairs:
            distance_indexes.append([self.topology.residue(ri).atom('CA').index,
                                     self.topology.residue(rj).atom('CA').index
                                     ])
        distance_indexes = np.array(distance_indexes)

        self.add_distances(distance_indexes, periodic=periodic)

    def add_inverse_distances(self, indices, periodic=True, indices2=None):
        """
        Adds the inverse distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the inverse distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the inverse distances shall be computed.

        periodic : optional, boolean, default is True
            If periodic is True and the trajectory contains unitcell information,
            distances will be computed under the minimum image convention.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the inverse distances between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.


        .. note::
            When using the *iterable of integers* input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.

        """
        from .distances import InverseDistanceFeature
        atom_pairs = _parse_pairwise_input(
            indices, indices2, self.logger, fname='add_inverse_distances()')

        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(self.topology, atom_pairs, periodic=periodic)
        self.__add_feature(f)

    def add_contacts(self, indices, indices2=None, threshold=0.3, periodic=True, count_contacts=False):
        r"""
        Adds the contacts to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the contacts shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the contacts shall be computed.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the contacts between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.

        threshold : float, optional, default = .3
            distances below this threshold (in nm) will result in a feature 1.0, distances above will result in 0.0.
            The default is set to .3 nm (3 Angstrom)

        periodic : boolean, default True
            use the minimum image convention if unitcell information is available

        count_contacts : boolean, default False
            If set to true, this feature will return the number of formed contacts (and not feature values with either 1.0 or 0)
            The ouput of this feature will be of shape (Nt,1), and not (Nt, nr_of_contacts)


        .. note::
            When using the *iterable of integers* input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.
        """
        from .distances import ContactFeature
        atom_pairs = _parse_pairwise_input(
            indices, indices2, self.logger, fname='add_contacts()')

        atom_pairs = self._check_indices(atom_pairs)
        f = ContactFeature(self.topology, atom_pairs, threshold, periodic, count_contacts)
        self.__add_feature(f)

    def add_residue_mindist(self,
                            residue_pairs='all',
                            scheme='closest-heavy',
                            ignore_nonprotein=True,
                            threshold=None,
                            periodic=True,
                            count_contacts=False):
        r"""
        Adds the minimum distance between residues to the feature list. See below how
        the minimum distance can be defined. If the topology generated out of :py:obj:`topfile`
        contains information on periodic boundary conditions, the minimum image convention
        will be used when computing distances.

        Parameters
        ----------
        residue_pairs : can be of two types:

            'all'
                Computes distances between all pairs of residues excluding first and second neighbors

            ndarray((n, 2), dtype=int):
                n x 2 array with the pairs residues for which distances will be computed

        scheme : 'ca', 'closest', 'closest-heavy', default is closest-heavy
                Within a residue, determines the sub-group atoms that will be considered when computing distances

        ignore_nonprotein : boolean, default True
                Ignore residues that are not of protein type (e.g. water molecules, post-traslational modifications etc)

        threshold : float, optional, default is None
            distances below this threshold (in nm) will result in a feature 1.0, distances above will result in 0.0. If
            left to None, the numerical value will be returned

        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.

        count_contacts : bool, optional, default = False
            If set to True, this feature will return the number of formed contacts (and not feature values with
            either 1.0 or 0). The ouput of this feature will be of shape (Nt,1), and not (Nt, nr_of_contacts).
            Requires threshold to be set.


        .. note::
            Using :py:obj:`scheme` = 'closest' or 'closest-heavy' with :py:obj:`residue pairs` = 'all'
            will compute nearly all interatomic distances, for every frame, before extracting the closest pairs.
            This can be very time consuming. Those schemes are intended to be used with a subset of residues chosen
            via :py:obj:`residue_pairs`.


        """
        from .distances import ResidueMinDistanceFeature
        if scheme != 'ca' and is_string(residue_pairs):
            if residue_pairs == 'all':
                self.logger.warning("Using all residue pairs with schemes like closest or closest-heavy is "
                                     "very time consuming. Consider reducing the residue pairs")

        f = ResidueMinDistanceFeature(self.topology, residue_pairs, scheme, ignore_nonprotein, threshold, periodic,
                                      count_contacts=count_contacts)
        self.__add_feature(f)

    def add_group_COM(self, group_definitions, ref_geom=None, image_molecules=False, mass_weighted=True,):
        r"""
        Adds the centers of mass (COM) in cartesian coordinates of a group or groups of atoms.
        If these group definitions coincide directly with residues, use :obj:`add_residue_COM` instead. No periodic
        boundaries are taken into account.

        Parameters
        ----------

        group_definitions : iterable of integers
            List of the groups of atom indices for which the COM will be computed. The atoms are zero-indexed.

        ref_geom : :obj:`mdtraj.Trajectory`, default is None
            The coordinates can be centered to a reference geometry before computing the COM.

        image_molecules : boolean, default is False
            The method traj.image_molecules will be called before computing averages. The method tries to correct
            for molecules broken across periodic boundary conditions, but can be time consuming. See
            http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules
            for more details

        mass_weighted : boolean, default is True
            Set to False if you want the geometric center and not the COM


        .. note::
            Centering (with :obj:`ref_geom`) and imaging (:obj:`image_molecules=True`) the trajectories can sometimes be time consuming. Consider doing that to your trajectory-files prior to the featurization.
        """

        from .misc import GroupCOMFeature

        f = GroupCOMFeature(self.topology, group_definitions , ref_geom=ref_geom, image_molecules=image_molecules, mass_weighted=mass_weighted)
        self.__add_feature(f)

    def add_residue_COM(self, residue_indices, scheme='all', ref_geom=None, image_molecules=False, mass_weighted=True,):
        r"""
        Adds a per-residue center of mass (COM) in cartesian coordinates.
        No periodic boundaries are taken into account.

        Parameters
        ----------

        residue_indices : iterable of integers
            The residue indices for which the COM will be computed. These are always zero-indexed that **are not
            necessarily** the residue sequence record of the topology (resSeq). resSeq indices start at least at 1
            but can depend on the topology. See http://mdtraj.org/latest/atom_selection.html for more details.

        scheme : str, default is 'all'
            What atoms contribute to the COM computation. The supported keywords are:
            'all', 'backbone', 'sidechain' . If the scheme yields no atoms for some residue, the selection
            falls back to 'all' for that residue.

        ref_geom : obj:`mdtraj.Trajectory`, default is None
            The coordinates can be centered to a reference geometry before computing the COM.

        image_molecules : boolean, default is False
            The method traj.image_molecules will be called before computing averages. The method tries to correct
            for molecules broken across periodic boundary conditions, but can be time consuming. See
            http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules
            for more details

        mass_weighted : boolean, default is True
            Set to False if you want the geometric center and not the COM


        .. note::
            Centering (with :obj:`ref_geom`) and imaging (:obj:`image_molecules=True`) the trajectories can sometimes be time consuming. Consider doing that to your trajectory-files prior to the featurization.
        """

        from .misc import ResidueCOMFeature
        from pyemma.coordinates.data.featurization.util import _atoms_in_residues
        assert scheme in ['all', 'backbone', 'sidechain']

        residue_atoms = _atoms_in_residues(self.topology, residue_indices, subset_of_atom_idxs=self.topology.select(scheme), MDlogger=self.logger)

        f = ResidueCOMFeature(self.topology, np.asarray(residue_indices), residue_atoms, scheme, ref_geom=ref_geom, image_molecules=image_molecules, mass_weighted=mass_weighted)

        self.__add_feature(f)

    def add_group_mindist(self, group_definitions, group_pairs='all', threshold=None,
                          periodic=True, count_contacts=False):
        r"""
        Adds the minimum distance between groups of atoms to the feature list. If the groups of
        atoms are identical to residues, use :py:obj:`add_residue_mindist <pyemma.coordinates.data.featurizer.MDFeaturizer.add_residue_mindist>`.

        Parameters
        ----------

        group_definitions : list of 1D-arrays/iterables containing the group definitions via atom indices.
            If there is only one group_definition, it is assumed the minimum distance within this group (excluding the
            self-distance) is wanted. In this case, :py:obj:`group_pairs` is ignored.

        group_pairs :  Can be of two types:
            'all'
                Computes minimum distances between all pairs of groups contained in the group definitions

            ndarray((n, 2), dtype=int):
                n x 2 array with the pairs of groups for which the minimum distances will be computed.

        threshold : float, optional, default is None
            distances below this threshold (in nm) will result in a feature 1.0, distances above will result in 0.0. If
            left to None, the numerical value will be returned

        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.

        count_contacts : bool, optional, default = False
            If set to True, this feature will return the number of formed contacts (and not feature values with
            either 1.0 or 0). The ouput of this feature will be of shape (Nt,1), and not (Nt, nr_of_contacts).
            Requires threshold to be set.

        """
        from .distances import GroupMinDistanceFeature
        # Some thorough input checking and reformatting
        group_definitions, group_pairs, distance_list, group_identifiers = \
            _parse_groupwise_input(group_definitions, group_pairs, self.logger, 'add_group_mindist')
        distance_list = self._check_indices(distance_list)

        f = GroupMinDistanceFeature(self.topology, group_definitions, group_pairs, distance_list, group_identifiers,
                                    threshold, periodic, count_contacts=count_contacts)
        self.__add_feature(f)

    def add_angles(self, indexes, deg=False, cossin=False, periodic=True):
        """
        Adds the list of angles to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 3), dtype=int
            an array with triplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.

        """
        from .angles import AngleFeature
        indexes = self._check_indices(indexes, pair_n=3)
        f = AngleFeature(self.topology, indexes, deg=deg, cossin=cossin,
                         periodic=periodic)
        self.__add_feature(f)

    def add_dihedrals(self, indexes, deg=False, cossin=False, periodic=True):
        """
        Adds the list of dihedrals to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 4), dtype=int
            an array with quadruplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.

        """
        from .angles import DihedralFeature
        indexes = self._check_indices(indexes, pair_n=4)
        f = DihedralFeature(self.topology, indexes, deg=deg, cossin=cossin,
                            periodic=periodic)
        self.__add_feature(f)

    def add_backbone_torsions(self, selstr=None, deg=False, cossin=False, periodic=True):
        """
        Adds all backbone phi/psi angles or the ones specified in :obj:`selstr` to the feature list.

        Parameters
        ----------

        selstr : str, optional, default = ""
            selection string specifying the atom selection used to specify a specific set of backbone angles
            If "" (default), all phi/psi angles found in the topology will be computed
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.
        """
        from .angles import BackboneTorsionFeature
        f = BackboneTorsionFeature(
            self.topology, selstr=selstr, deg=deg, cossin=cossin, periodic=periodic)
        self.__add_feature(f)

    @deprecated('Please use "add_sidechain_torsions(which=[\'chi1\'])"')
    def add_chi1_torsions(self, selstr="", deg=False, cossin=False, periodic=True):
        """
        Adds all chi1 angles or the ones specified in :obj:`selstr` to the feature list.

        Parameters
        ----------

        selstr : str, optional, default = ""
            selection string specifying the atom selection used to specify a specific set of backbone angles
            If "" (default), all chi1 angles found in the topology will be computed
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        periodic : bool, optional, default = True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.
        """
        from .angles import SideChainTorsions
        f = SideChainTorsions(
            self.topology, selstr=selstr, deg=deg, cossin=cossin, periodic=periodic, which=['chi1'])
        self.__add_feature(f)

    def add_sidechain_torsions(self, selstr=None, deg=False, cossin=False, periodic=True, which='all'):
        """
        Adds all side chain torsion angles or the ones specified in :obj:`selstr` to the feature list.

        Parameters
        ----------
        selstr: str, optional, default=None
            selection string specifying the atom selection used to specify a specific set of backbone angles
            If "" (default), all chi1 angles found in the topology will be computed
        deg: bool, optional, default=False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin: bool, optional, default=False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        periodic: bool, optional, default=True
            If `periodic` is True and the trajectory contains unitcell
            information, we will treat dihedrals that cross periodic images
            using the minimum image convention.
        which: str or list of str, default='all'
            one or combination of ('all', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5')
        """
        from .angles import SideChainTorsions
        f = SideChainTorsions(self.topology, selstr=selstr, deg=deg, cossin=cossin, periodic=periodic, which=which)
        self.__add_feature(f)

    def add_custom_feature(self, feature):
        """
        Adds a custom feature to the feature list.

        Parameters
        ----------
        feature : object
            an object with interface like CustomFeature (map, describe methods)

        """
        if feature.dimension <= 0:
            raise ValueError("Dimension has to be positive. "
                             "Please override dimension attribute in feature!")

        if not hasattr(feature, 'transform'):
            raise ValueError("no 'transform' method in given feature")
        elif not callable(getattr(feature, 'transform')):
            raise ValueError("'transform' attribute exists but is not a method")

        self.__add_feature(feature)

    def add_minrmsd_to_ref(self, ref, ref_frame=0, atom_indices=None, precentered=False):
        r"""
        Adds the minimum root-mean-square-deviation (minrmsd) with respect to a reference structure to the feature list.

        Parameters
        ----------
        ref:
            Reference structure for computing the minrmsd. Can be of two types:

                1. :py:obj:`mdtraj.Trajectory` object
                2. filename for mdtraj to load. In this case, only the :py:obj:`ref_frame` of that file will be used.

        ref_frame: integer, default=0
            Reference frame of the filename specified in :py:obj:`ref`.
            This parameter has no effect if :py:obj:`ref` is not a filename.

        atom_indices: array_like, default=None
            Atoms that will be used for:

                1. aligning the target and reference geometries.
                2. computing rmsd after the alignment.
            If left to None, all atoms of :py:obj:`ref` will be used.

        precentered: bool, default=False
            Use this boolean at your own risk to let mdtraj know that the target conformations are already
            centered at the origin, i.e., their (uniformly weighted) center of mass lies at the origin.
            This will speed up the computation of the rmsd.
        """
        from .misc import MinRmsdFeature
        f = MinRmsdFeature(ref, ref_frame=ref_frame, atom_indices=atom_indices, topology=self.topology,
                           precentered=precentered)
        self.__add_feature(f)

    def add_custom_func(self, func, dim, *args, **kwargs):
        """ adds a user defined function to extract features

        Parameters
        ----------
        func : function
            a user-defined function, which accepts mdtraj.Trajectory object as
            first parameter and as many optional and named arguments as desired.
            Has to return a numpy.ndarray ndim=2.
        dim : int
            output dimension of :py:obj:`function`
        description: str or None
            a message for the describe feature list.
        args : any number of positional arguments
            these have to be in the same order as :py:obj:`func` is expecting them
        kwargs : dictionary
            named arguments passed to func

        Notes
        -----
        You can pass a description list to describe the output of your function by element,
        by passing a list of strings with the same lengths as dimensions.
        Alternatively a single element list or str will be expanded to match the output dimension.

        """
        description = kwargs.pop('description', None)
        f = CustomFeature(func, dim=dim, description=description, fun_args=args, fun_kwargs=kwargs)
        self.add_custom_feature(f)

    def remove_all_custom_funcs(self):
        """ Remove all instances of CustomFeature from the active feature list.
        """
        custom_feats = [f for f in self.active_features if isinstance(f, CustomFeature)]
        for f in custom_feats:
            self.active_features.remove(f)

    def dimension(self):
        """ current dimension due to selected features

        Returns
        -------
        dim : int
            total dimension due to all selection features

        """
        dim = sum(f.dimension for f in self.active_features)
        return dim

    def transform(self, traj):
        """
        Maps an mdtraj Trajectory object to the selected output features

        Parameters
        ----------
        traj : mdtraj Trajectory
            Trajectory object used as an input

        Returns
        -------
        out : ndarray((T, n), dtype=float32)
            Output features: For each of T time steps in the given trajectory,
            a vector with all n output features selected.

        """
        # if there are no features selected, return given trajectory
        if not self.active_features:
            self.add_selection(np.arange(self.topology.n_atoms))
            warnings.warn("You have not selected any features. Returning plain coordinates.")

        # otherwise build feature vector.
        feature_vec = []

        # TODO: consider parallel evaluation computation here, this effort is
        # only worth it, if computation time dominates memory transfers
        for f in self.active_features:
            # perform sanity checks for custom feature input
            if isinstance(f, CustomFeature):
                # NOTE: casting=safe raises in numpy>=1.9
                vec = f.transform(traj).astype(np.float32, casting='safe')
                if vec.shape[0] == 0:
                    vec = np.empty((0, f.dimension))

                if not isinstance(vec, np.ndarray):
                    raise ValueError('Your custom feature %s did not return'
                                     ' a numpy.ndarray!' % str(f.describe()))
                if not vec.ndim == 2:
                    raise ValueError('Your custom feature %s did not return'
                                     ' a 2d array. Shape was %s'
                                     % (str(f.describe()),
                                        str(vec.shape)))
                if not vec.shape[0] == traj.xyz.shape[0]:
                    raise ValueError('Your custom feature %s did not return'
                                     ' as many frames as it received!'
                                     'Input was %i, output was %i'
                                     % (str(f.describe()),
                                        traj.xyz.shape[0],
                                        vec.shape[0]))
            else:
                vec = f.transform(traj).astype(np.float32)
            feature_vec.append(vec)

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]

        return res
