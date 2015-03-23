__author__ = 'noe'

import numpy as np
import mdtraj

from pyemma.coordinates.util import patches
from pyemma.coordinates.io.reader import ChunkedReader
from pyemma.util.log import getLogger

from .featurizer import MDFeaturizer

log = getLogger('FeatureReader')

__all__ = ['FeatureReader']


class FeatureReader(ChunkedReader):

    """
    Reads features from MD data.

    To select a feature, access the :attr:`featurizer` and call a feature
    selecting method (e.g) distances.

    Parameters
    ----------
    trajectories: list of strings
        paths to trajectory files

    topologyfile: string
        path to topology file (e.g. pdb)

    Examples
    --------

    Iterator access:

    >>> reader = FeatureReader('mytraj.xtc', 'my_structure.pdb')
    >>> chunks = []
    >>> for itraj, X in reader:
    >>>     chunks.append(X)


    Extract backbone torsion angles of protein during feature reading:

    >>> reader = FeatureReader('mytraj.xtc', 'my_structure.pdb')
    >>> reader.featurizer.backbone_torsions()
    >>> chunks = []
    >>> for _, X in reader:
    ...     chunks.append(X)

    """

    def __init__(self, trajectories, topologyfile):
        # init with chunksize 100
        ChunkedReader.__init__(self, 100)

        # files
        if isinstance(trajectories, str):
            trajectories = [trajectories]
        self.trajfiles = trajectories
        self.topfile = topologyfile

        # featurizer
        self.featurizer = MDFeaturizer(topologyfile)

        # _lengths
        self._lengths = []
        self._totlength = 0

        # iteration
        self.mditer = None
        # current lag time
        self.curr_lag = 0
        # time lagged iterator
        self.mditer2 = None

        # cache size
        self.in_memory = False
        self.Y = None
        # basic statistics
        for traj in trajectories:
            sum_frames = sum(t.n_frames for t in self._create_iter(traj))
            self._lengths.append(sum_frames)

        self._totlength = np.sum(self._lengths)

        self.t = 0

    def describe(self):
        """
        Returns a description of this transformer

        :return:
        """
        return "Feature reader, features = ", self.featurizer.describe()

    def operate_in_memory(self):
        """
        If called, the output will be fully stored in memory

        :return:
        """
        self.in_memory = True
        # output data
        self.Y = [np.empty((self.trajectory_length(itraj), self.dimension()))
                  for itraj in xrange(self.number_of_trajectories())]

    def parametrize(self):
        """
        Parametrizes this transformer

        :return:
        """
        if self.in_memory:
            self.map_to_memory()

    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return len(self.trajfiles)

    def trajectory_length(self, itraj):
        """
        Returns the length of trajectory

        Parameters
        ----------
        itraj : int

        :return:
            length of trajectory
        """
        return self._lengths[itraj]

    def trajectory_lengths(self):
        """
        Returns the trajectory _lengths in a list
        :return:
        """
        return self._lengths

    def n_frames_total(self):
        """
        Returns the total number of frames, summed over all trajectories
        :return:
        """
        return self._totlength

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self.featurizer.dimension()

    def get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes

        :return:
        """
        return 4 * self.dimension()

    def get_constant_memory(self):
        """
        Returns the constant memory requirements, in bytes

        :return:
        """
        return 0

    def map_to_memory(self):
        self.reset()
        # iterate over trajectories
        last_chunk = False
        itraj = 0
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                y = self.next_chunk()
                assert y is not None
                L = np.shape(y)[0]
                # last chunk in traj?
                last_chunk_in_traj = (t + L >= self.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                # write
                self.Y[itraj][t:t + L] = y
                # increment time
                t += L
            # increment trajectory
            itraj += 1

    def _create_iter(self, filename, skip=0):
        return patches.iterload(filename, chunk=self.chunksize,
                                top=self.topfile, skip=skip)

    def reset(self):
        """
        resets the chunk reader
        """
        self.curr_itraj = 0
        self.curr_lag = 0
        if len(self.trajfiles) >= 1:
            self.t = 0
            self.mditer = self._create_iter(self.trajfiles[0])

    def next_chunk(self, lag=0):
        """
        gets the next chunk. If lag > 0, we open another iterator with same chunk
        size and advance it by one, as soon as this method is called with a lag > 0.

        :return: a feature mapped vector X, or (X, Y) if lag > 0
        """
        chunk = self.mditer.next()

        if lag > 0:
            if self.curr_lag == 0:
                # lag time or trajectory index changed, so open lagged iterator
                log.debug("open time lagged iterator for traj %i with lag %i"
                          % (self.curr_itraj, self.curr_lag))
                self.curr_lag = lag
                self.mditer2 = self._create_iter(self.trajfiles[self.curr_itraj],
                                                skip=self.curr_lag)
            try:
                adv_chunk = self.mditer2.next()
            except StopIteration:
                # When mditer2 ran over the trajectory end, return empty chunks.
                adv_chunk = mdtraj.Trajectory(
                              np.empty((0,chunk.xyz.shape[1],chunk.xyz.shape[2]),np.float32),
                              chunk.topology)

        self.t += chunk.xyz.shape[0]

        if (self.t >= self.trajectory_length(self.curr_itraj) and
                self.curr_itraj < len(self.trajfiles) - 1):
            log.debug('closing current trajectory "%s"'
                      % self.trajfiles[self.curr_itraj])
            self.mditer.close()
            self.t = 0
            self.curr_itraj += 1
            self.mditer = self._create_iter(self.trajfiles[self.curr_itraj])
            # we open self.mditer2 only if requested due lag parameter!
            self.curr_lag = 0

        # map data
        if lag == 0:
            return self.featurizer.map(chunk)
        else:
            # TODO: note X and Y may have different shapes. This case has to be handled by subsequent transformers
            X = self.featurizer.map(chunk)
            Y = self.featurizer.map(adv_chunk)
#             assert np.shape(X) == np.shape(Y), "shape X = %s; Y= %s; lag=%i" % (
#                 np.shape(X), np.shape(Y), lag)
            return X, Y

    def __iter__(self):
        self.reset()
        return self

    def next(self):
        """ enable iteration over transformed data.

        Returns
        -------
        (itraj, X) : (int, ndarray(n, m)
            itraj corresponds to input sequence number (eg. trajectory index)
            and X is the transformed data, n = chunksize or n < chunksize at end
            of input.

        """
        # iterate over trajectories
        if self.curr_itraj >= self.number_of_trajectories():
            raise StopIteration

        # next chunk already maps output
        if self.lag == 0:
            X = self.next_chunk()
        else:
            X, Y = self.next_chunk(self.lag)
            ## we wont be able to correlate chunks of different len, so stop here
            #if np.shape(X) != np.shape(Y):
            #    # TODO: determine if its possible to truncate X to shape of Y?
            #    raise StopIteration

        last_itraj = self.curr_itraj
        # note: t is incremented in next_chunk
        if self.t >= self.trajectory_length(self.curr_itraj):
            self.curr_itraj += 1
            self.t = 0

        if self.lag == 0:
            return (last_itraj, X)

        return (last_itraj, X, Y)


