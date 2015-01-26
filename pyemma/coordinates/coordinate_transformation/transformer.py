__author__ = 'noe'

import numpy as np


class Transformer(object):

    # param finished?
    param_finished = False
    # data_producer
    data_producer = None

    def set_data_producer(self, data_producer):
        """
        Sets the data producer for this transformation

        :param data_producer:
        :return:
        """
        self.data_producer = data_producer


    def set_chunksize(self, size):
        self.chunksize = size


    def operate_in_memory(self):
        """
        If called, the output will be stored in memory
        :return:
        """
        self.in_memory = True
        # output data
        self.Y = [np.zeros((self.trajectory_length(itraj), self.dimension())) for itraj in range(0,self.number_of_trajectories())]


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
        # check if ready
        if self.data_producer is None:
            raise RuntimeError('Called parametrize while data producer is not yet set. Call set_data_producer first!')
        # init
        self.param_init()
        # feed data, until finished
        add_data_finished = False
        ipass = 0
        lag = self.get_lag()
        # parametrize
        while add_data_finished != True:
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
                        X, Y = self.data_producer.next_chunk(lag=lag)
                    L = np.shape(X)[0]
                    # last chunk in traj?
                    last_chunk_in_traj = (t + lag + L >= self.trajectory_length(itraj))
                    # last chunk?
                    last_chunk = (last_chunk_in_traj and itraj >= self.number_of_trajectories()-1)
                    # first chunk
                    add_data_finished = self.param_add_data(X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=Y)
                    first_chunk = False
                    # increment time
                    t += L
                # increment trajectory
                itraj += 1
            ipass += 1
        # finish parametrization
        self.param_finish()
        # memory mode? Then map all results
        if self.in_memory:
            self.map_to_memory()
        # done!
        self.param_finished = True


    def parametrized(self):
        return self.param_finished

    def param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        # create mean array and covariance matrix
        pass


    def param_finish(self):
        """
        Finalizes the parametrization.

        :return:
        """
        pass

    def map_to_memory(self):
        # if operating in main memory, do all the mapping now
        self.data_producer.reset()
        # iterate over trajectories
        last_chunk = False
        itraj = 0
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                X = self.data_producer.next_chunk()
                L = np.shape(X)[0]
                # last chunk in traj?
                last_chunk_in_traj = (t + L >= self.trajectory_length(itraj))
                # last chunk?
                last_chunk = (last_chunk_in_traj and itraj >= self.number_of_trajectories()-1)
                # write
                self.Y[itraj][t:t+L] = self.map(X)
                # increment time
                t += L
            # increment trajectory
            itraj += 1


    def reset(self):
        if self.in_memory:
            # operate in memory, implement iterator here
            self.itraj = 0
            self.t = 0
        else:
            # operate in pipeline
            self.data_producer.reset()


    def next_chunk(self, lag = 0):
        if self.in_memory:
            if self.itraj >= self.number_of_trajectories():
                return None
            # operate in memory, implement iterator here
            if lag == 0:
                Y = self.Y[self.itraj][self.t:min(self.t+self.chunksize,self.trajectory_length(self.itraj))]
                # increment counters
                self.t += self.chunksize
                if self.t >= self.trajectory_length(self.itraj):
                    self.itraj += 1
                    self.t = 0
                return Y
            else:
                Y0 = self.Y[self.itraj][self.t:min(self.t+self.chunksize,self.trajectory_length(self.itraj))]
                Ytau = self.Y[self.itraj][self.t+lag:min(self.t+self.chunksize+lag,self.trajectory_length(self.itraj))]
                # increment counters
                self.t += self.chunksize
                if self.t >= self.trajectory_length(self.itraj):
                    self.itraj += 1
                return (Y0,Ytau)
        else:
            # operate in pipeline
            if lag == 0:
                X = self.data_producer.next_chunk()
                return self.map(X)
            else:
                (X0,Xtau) = self.data_producer.next_chunk(lag = lag)
                return (self.map(X0),self.map(Xtau))


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
        return np.linalg.norm(Y - x, 2, axis=1)
