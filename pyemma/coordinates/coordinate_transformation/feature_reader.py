__author__ = 'noe'

import mdtraj
import numpy as np

class FeatureReader:
    """
    Reads features from MD data
    """

    # files
    trajfiles = []
    topfile = None
    featurizer = None

    # lengths
    lengths = []
    totlength = 0

    # iteration
    mditer = None
    curr_itraj = 0

    # cache size
    chunksize = 10000
    in_memory = False
    Y = None


    def __init__(self, trajectories, topologyfile, featurizer):
        """
        Constructs a feature reader

        :param trajectories:
            list of trajectory files

        :param structurefile:
            structure file (e.g. pdb)

        """
        # init
        self.trajfiles = trajectories
        self.topfile = topologyfile
        self.featurizer = featurizer

        # basic statistics
        for traj in trajectories:
            t = mdtraj.open(traj)
            self.lengths.append(len(t))
        self.totlength = np.sum(self.lengths)

        # load first trajectory
        self.curr_traj = mdtraj.open(trajectories[0])


    def describe(self):
        return "Feature reader, features = ",self.featurizer.describe()


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
            self.curr_traj = mdtraj.load(self.trajfiles[itraj], top=self.topfile)
            self.curr_itraj = itraj
        frame = self.curr_traj.slice(t)
        return self.featurizer.map(frame)


    def reset(self):
        """
        resets the chunk reader
        :return:
        """
        self.curr_itraj = 0
        self.mditer = mdtraj.iterload(self.trajfiles[0], chunk=self.chunksize, top=self.topfile)


    # TODO: enable iterating over lagged pairs of chunks!
    def next_chunk(self):
        """
        gets the next chunk

        :return:
        """
        chunk = self.mditer.next()

        if np.max(chunk.time) >= self.trajectory_length(self.curr_itraj)-1:
            self.mditer.close()
            if self.curr_itraj < len(self.trajfiles)-1:
                self.curr_itraj += 1
                self.mditer = mdtraj.iterload(self.trajfiles[self.curr_itraj], chunk=self.chunksize, top=self.topfile)

        return self.featurizer.map(chunk)