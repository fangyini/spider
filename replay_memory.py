import numpy as np
from utils import Paras
import random
import torch

parameters = Paras()
random.seed(parameters.seed)


class ReplayMemory:
    """
    Replay memory storing experience tuples from previous episodes. Once can
    obtain batches of experience tuples from the replay memory for training
    a policy network. Old tuples are overwritten once memory limit is
    reached.
    """

    def __init__(self, size, column_types, column_shapes=None, batch_size=64, gpuid=None):
        """
        :param size: Number of experience tuples to be stored in the replay
        memory.
        :param column_types: Dictionary mapping column names to data type of
        data to be stored in it.
        :param column_shapes: Shapes of the data stored in the cells of each
        column as lists. Empty list for scalar values. If None, all shapes are
        considered scalar.
        :param batch_size: Size of the batch which is sampled from the replay
        memory.
        """

        self.size = size
        self.batch_size = batch_size
        self.column_types = column_types

        # Convert shape of individual rows to shape of entire columns.
        if column_shapes is None:
            column_shapes = {key: [] for key in column_types.keys()}
        self.column_shapes = {key: [self.size]+shape for (key, shape)
                         in column_shapes.items()}

        # Preallocate memory
        self.columns = {}
        for column_name, data_type in column_types.items():
            self.columns[column_name] = torch.empty(self.column_shapes[column_name],
                                                 dtype=data_type).to(gpuid)

        # Index of the row in the replay memory which gets replaced during the
        # next insert.
        self.current = 0
        # Total number of rows stored in the replay memory.
        self.count = 0

    def save_first_20(self):
        num = 5000
        indices = random.sample(range(self.count), num)
        for column_name, data_type in self.column_types.items():
            self.columns[column_name][:num] = self.columns[column_name][indices]
        self.current = num
        self.count = num


    def add_all(self, rows):
        """
        Add multiple new rows of data to replay memory.
        :param rows: Dictionary of lists containing the new rows' data for each
        column.
        """
        # assert len(actions) == len(rewards) == len(obs) == len(values)
        num = len(list(rows.values())[0])
        assert all(len(x) == num for x in rows.values())

        if self.current + num <= self.size:
            for column_name in self.columns.keys():
                #a = self.columns[column_name][np.arange(num)+self.current]
                #b = rows[column_name]
                self.columns[column_name][torch.arange(num)+self.current] = \
                    rows[column_name]
        else:
            num_free = self.size - self.current
            num_over = num - num_free
            # Insert first few elements at the end
            for column_name in self.columns.keys():
                self.columns[column_name][self.current:] = \
                    rows[column_name][:num_free]
            # Insert remaining elements at the front
            for column_name in self.columns.keys():
                self.columns[column_name][:num_over] = \
                    rows[column_name][num_free:]

        self.count = max(self.count, min(self.current + num, self.size))
        self.current = (self.current + num) % self.size

    def shuffle(self):
        ind = [torch.randperm(self.count)]
        for column_name in self.columns.keys():
            self.columns[column_name][:self.count] = self.columns[column_name][ind]

    def get_minibatch(self, start, end):
        """
        Returns a batch of experience tuples for training.
        :return: Dictionary containing a numpy array for each column.
        """
        #indices = random.sample(range(self.count), self.batch_size)
        indices = torch.arange(start, end)
        minibatch = {}
        for column_name in self.columns.keys():
            minibatch[column_name] = self.columns[column_name][indices]
        return minibatch

    def get_minibatch(self):
        """
        Returns a batch of experience tuples for training.
        :return: Dictionary containing a numpy array for each column.
        """
        indices = random.sample(range(self.count), self.batch_size)
        minibatch = {}
        for column_name in self.columns.keys():
            minibatch[column_name] = self.columns[column_name][indices]
        return minibatch

    def __str__(self):
        descr = ""
        for column_name in self.columns.keys():
            descr += "{0}: {1}\n".format(column_name,
                                         self.columns[column_name].__str__())
        return descr

    def save_memory(self, filename):
        obs = self.columns['ob'][:self.count]
        act = self.columns['act'][:self.count]
        ret = self.columns['time'][:self.count]
        torch.save(obs, filename + '_obs.pt')
        torch.save(act, filename + '_act.pt')
        torch.save(ret, filename + '_time.pt')
        #np.save(filename, self.columns)