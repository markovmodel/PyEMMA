from mdtraj.core.trajectory import Trajectory
__author__ = 'noe'

import mdtraj
import numpy as np


class FeatureReader:

    """
    Reads features from MD data
    """

    def __init__(self, trajectories, topologyfile, featurizer):
        """
        Constructs a feature reader

        :param trajectories:
            list of trajectory files

        :param structurefile:
            structure file (e.g. pdb)

        """
        # files
        self.trajfiles = trajectories
        self.topfile = topologyfile
        self.featurizer = featurizer

        # lengths
        self.lengths = []
        self.totlength = 0

        # iteration
        self.mditer = None
        # current lag time
        self.curr_lag = 0
        # time lagged iterator
        self.mditer2 = None
        self.curr_itraj = 0

        # cache size
        self.chunksize = 10000
        self.in_memory = False
        self.Y = None
        Trajectory
        # basic statistics
        for traj in trajectories:
            print "determing length of traj '%s'..." % traj
            sum_frames = sum(t.n_frames for t in
                             mdtraj.iterload(traj, top=self.topfile,
                                             chunk=self.chunksize))
            print "finished"
            self.lengths.append(sum_frames)

        print "len of trajectories:", self.lengths
        self.totlength = np.sum(self.lengths)

        # load first trajectory
        self.curr_traj = mdtraj.open(trajectories[0])

    def describe(self):
        return "Feature reader, features = ", self.featurizer.describe()

    def set_chunksize(self, size):
        self.chunksize = size

    def operate_in_memory(self):
        """
        If called, the output will be stored in memory
        :return:
        """
        self.in_memory = True

    def parametrize(self):
        pass

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

        :param itraj:
            trajectory index

        :return:
            length of trajectory
        """
        return self.lengths[itraj]

    def trajectory_lengths(self):
        return self.lengths

    def n_frames_total(self):
        return self.totlength

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

    def get(self, itraj, t):
        """
        Returns
        :param itraj:
        :param t:
        :return:
        """
        if (itraj != self.curr_itraj):
            self.curr_traj = mdtraj.load(
                self.trajfiles[itraj], top=self.topfile)
            self.curr_itraj = itraj
        frame = self.curr_traj.slice(t)
        return self.featurizer.map(frame)

    def reset(self):
        """
        resets the chunk reader
        :return:
        """
        self.curr_itraj = 0
        if len(self.trajfiles) >= 1:
            self.mditer = mdtraj.iterload(self.trajfiles[0],
                                          chunk=self.chunksize, top=self.topfile)

            if self.curr_lag > 0:
                self.mditer2 = mdtraj.iterload(self.trajfiles[0],
                                               chunk=self.chunksize, top=self.topfile)
                # forward iterator
                def consume(iterator, n):
                    '''Advance the iterator n-steps ahead. If n is none, consume entirely.'''
                    import collections
                    import itertools
                    collections.deque(itertools.islice(iterator, n), maxlen=0)

                # for
                try:
                    consume(self.mditer2, self.curr_lag)
                except StopIteration:
                    raise RuntimeError("lag time might be bigger than trajectory length!")

    # TODO: enable iterating over lagged pairs of chunks!
    def next_chunk(self, lag=0):
        """
        gets the next chunk

        :return:
        """
        chunk = self.mditer.next()

        if lag > 0:
            # lag time changed
            self.curr_lag = lag

        if np.max(chunk.time) >= self.trajectory_length(self.curr_itraj) - 1:
            self.mditer.close()
            if self.curr_itraj < len(self.trajfiles) - 1:
                self.curr_itraj += 1
                self.mditer = mdtraj.iterload(
                    self.trajfiles[self.curr_itraj], chunk=self.chunksize, top=self.topfile)
        if self.curr_lag == 0:
            return self.featurizer.map(chunk)
        else:
            chunk2 = self.mditer2.next()
            return self.featurizer.map(chunk, chunk2)
