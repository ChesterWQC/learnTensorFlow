"""

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


class RegressionTrainingSet(object):
    """
    Training set including features and the result
    """
    def __init__(self, data=None):
        """

        :param data: 2-d list
        """
        if data is None:
            return
        self._data_set = np.array(data)
        self.sample_size = len(data)
        self.feature_number = len(data[0]) - 1
        self.features = self._data_set[:, :self.feature_number]  # numpy matrix [[1,2,3],[4,5,6]]
        self.normalized_features = self.features
        self.normalized = False
        self.results = self._data_set[:, self.feature_number:]  # numpy column vector
        self.mean = np.mean(self.features, axis=0)\
            .reshape(1, self.feature_number)  # numpy row vector
        self.sigma = np.std(self.features, axis=0, ddof=1)\
            .reshape(1, self.feature_number)  # numpy row vector
        self.init_theta = np.zeros((1, self.feature_number + 1))  # numpy row vector
        self.final_theta = self.init_theta  # numpy row vector
        self.trained = False
        self.cost_history = []

    @classmethod
    def read_from_txt(cls, path):
        with open(path, 'r') as f:
            data = []
            for line in f:
                if line.endswith('\n'):
                    line = line[:len(line) - 1]
                line = line.split(",")
                arr = [float(i) for i in line]
                data.append(arr)
        return cls(data)

    @classmethod
    def read_from_mat(cls, path):
        contents = sio.loadmat(path)
        model = cls()
        x = contents['X']
        y = contents['y']
        model._data_set = np.hstack((x, y))
        model.sample_size = x.shape[0]
        model.feature_number = x.shape[1]
        model.features = x  # numpy matrix [[1,2,3],[4,5,6]]
        model.normalized_features = model.features
        model.normalized = False
        model.results = y  # numpy column vector
        model.mean = np.mean(model.features, axis=0) \
            .reshape(1, model.feature_number)  # numpy row vector
        model.sigma = np.std(model.features, axis=0, ddof=1) \
            .reshape(1, model.feature_number)  # numpy row vector
        model.init_theta = np.zeros((1, model.feature_number + 1))  # numpy row vector
        model.final_theta = model.init_theta
        model.trained = False
        model.cost_history = []
        return model

    def normalize(self):
        """

        :return: normalized features matrix (numpy matrix)
        """
        if self.normalized:
            return self.normalized_features
        else:
            self.normalized_features = (self.features - self.mean) / self.sigma
            self.normalized = True
            return self.normalized_features

    def predict(self, new_feature):
        pass

    def train_with_gd(self, alpha=0.01, iterations=10000, plot_cost=False,
                      regularization=False, lbd=0):
        pass

    def train_with_op(self, normalization=True, regularization=False, lbd=0):
        pass

    def clear_trained_result(self):
        self.final_theta = self.init_theta
        self.trained = False
        self.cost_history = []

    def plot_cost(self):
        plt.plot(self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("J")
        plt.show()









