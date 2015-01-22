__author__ = 'noe'

import mdtraj
import numpy as np

from mdtraj.core.trajectory import Trajectory



class FeatureReader(object):

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

        # cache size
        self.chunksize = 10000
        self.in_memory = False
        self.Y = None
        # basic statistics
        for traj in trajectories:
            sum_frames = sum(t.n_frames for t in
                             mdtraj.iterload(traj, top=self.topfile,
                                             chunk=self.chunksize))
            self.lengths.append(sum_frames)

        self.totlength = np.sum(self.lengths)

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
        self.Y = [np.zeros((self.trajectory_length(itraj), self.dimension()))
                  for itraj in range(0, self.number_of_trajectories())]

    def parametrize(self):
        """
        Parametrizes this transformer

        :return:
        """
        if self.in_memory:
            self.map_to_memory()
        self.param_finished = True


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
        """
        Returns the trajectory lengths in a list
        :return:
        """
        return self.lengths


    def n_frames_total(self):
        """
        Returns the total number of frames, summed over all trajectories
        :return:
        """
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


    def parametrized(self):
        """
        Returns whether the parametrization is finished

        :return: True if parametrization is finished
        """
        return self.param_finished


    def _open_time_lagged(self):
        if self.mditer2 is not None:
            self.mditer2.close()
        self.mditer2 = mdtraj.iterload(self.trajfiles[0],
                                       chunk=self.chunksize, top=self.topfile)


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
                self._open_time_lagged()


    def next_chunk(self, lag=0):
        """
        gets the next chunk. If lag > 0, we open another iterator with same chunk
        size and advance it by one, as soon as this method is called with a lag > 0.

        :return: a feature mapped vector X, or X, Y if lag > 0
        """
        chunk = self.mditer.next()

        if lag > 0:
            advanced_once = True
            if self.curr_lag == 0:
                # lag time changed
                self.curr_lag = lag
                self._open_time_lagged()
                # advance second iterator by one chunk
                try:
                    self.mditer2.next()
                except StopIteration:
                    advanced_once = False
            try:
                assert self.mditer2
                if advanced_once:  # we were able to increment mditer2 once
                    adv_chunk = self.mditer2.next()
            except StopIteration:
                # no more data available in mditer2, so we have to take data from
                # last_chunk and padd it with zeros!
                lagged_xyz = np.zeros_like(chunk.xyz)
                lagged_xyz[0:-lag] = chunk.xyz[lag:]
            else:
                # build time lagged Trajectory from current and advanced chunk
                lagged_xyz = np.empty_like(chunk.xyz)
                lagged_xyz[:-lag] = chunk.xyz[lag:]
                lagged_xyz[-lag:] = adv_chunk.xyz[0:lag]
            chunk_lagged = Trajectory(lagged_xyz, chunk.topology)

        if np.max(chunk.time) >= self.trajectory_length(self.curr_itraj) - 1:
            if self.curr_itraj < len(self.trajfiles) - 1:
                self.mditer.close()
                self.curr_itraj += 1
                self.mditer = mdtraj.iterload(
                    self.trajfiles[self.curr_itraj], chunk=self.chunksize, top=self.topfile)
                # we open self.mditer2 only if requested due lag parameter!

        # map data
        if lag == 0:
            return self.featurizer.map(chunk)
        else:
            X = self.featurizer.map(chunk)
            Y = self.featurizer.map(chunk_lagged)
            return X, Y
