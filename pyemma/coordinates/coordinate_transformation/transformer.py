__author__ = 'noe'

import numpy as np

class Transformer:

    def set_chunksize(self, size):
        self.chunksize = size

    def operate_in_memory(self):
        """
        If called, the output will be stored in memory
        :return:
        """
        self.in_memory = True

    def get_lag(self):
        """
        Returns 0 by default
        :return:
        """
        return 0


    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return self.data_producer.number_of_trajectories()


    def trajectory_length(self, itraj):
        """
        Returns the length of trajectory

        :param itraj:
            trajectory index

        :return:
            length of trajectory
        """
        return self.data_producer.trajectory_length(itraj)


    def trajectory_lengths(self):
        """
        Returns the length of each trajectory

        :return:
            length of each trajectory
        """
        return self.data_producer.trajectory_lengths()



    def n_frames_total(self):
        return self.data_producer.n_frames_total()


    def get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes

        :return:
        """
        return 4 * self.dimension()


    def parametrize(self):
        ipass = 0
        lag = self.get_lag()
        while not self.parametrization_finished():
            first_chunk = True
            self.data_producer.reset()
            # iterate over trajectories
            last_chunk = False
            itraj = 0
            while not last_chunk:
                last_chunk_in_traj = False
                t = 0
                while not last_chunk_in_traj:
                    # iterate over times within trajectory
                    if lag == 0:
                        X = self.data_producer.next_chunk()
                        Y = None
                    else:
                        (X,Y) = self.data_producer.next_chunk(lag=lag)
                    L = np.shape(X)[0]
                    # last chunk in traj?
                    last_chunk_in_traj = (t + lag + L >= self.trajectory_length(itraj))
                    # last chunk?
                    last_chunk = (last_chunk_in_traj and itraj >= self.number_of_trajectories()-1)
                    # first chunk
                    if Y is None:
                        self.add_chunk(X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass)
                    else:
                        self.add_chunk(X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=Y)
                    first_chunk = False
                    # increment time
                    t += L
                # increment trajectory
                itraj += 1
            ipass += 1



    def reset(self):
        self.data_producer.reset()


    def next_chunk(self, lag = 0):
        if lag == 0:
            X = self.data_producer.next_chunk()
            return self.map(X)
        else:
            (X,Y) = self.data_producer.next_chunk(lag = lag)
            return (self.map(X),self.map(Y))


    def distance(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return np.linalg.norm(x-y, 2)

    def distances(self, x, Y):
        """

        :param x: ndarray (n)
        :param y: ndarray (Nxn)
        :return:
        """
        np.linalg.norm(Y - x, 2, axis=1)
