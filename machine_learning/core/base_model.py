"""

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


class TrainingType:

    LINEAR_REGRESSION = 'LINEAR_REGRESSION'
    LOGISTIC_REGRESSION_BINARY = 'LOGISTIC_REGRESSION_BINARY'
    LOGISTIC_REGRESSION_MULTI = 'LOGISTIC_REGRESSION_MULTI'

    def __init__(self):
        pass


class RegressionTrainingSet(object):
    """
    Training set including features and the result
    """
    def __init__(self, x, y):
        """

        :param x: m * n matrix
        :param y: m * 1 matrix
        """

        self._data_set = np.hstack((x, y))
        self.sample_size = x.shape[0]
        self.feature_number = x.shape[1]
        self.features = x  # numpy matrix [[1,2,3],[4,5,6]]
        self.normalized_features = self.features
        self.normalized = False
        self.results = y  # numpy column vector
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
        data_set = np.array(data)
        x = data_set[:, :data_set.shape[1] - 1]
        y = data_set[:, data_set.shape[1] - 1:]
        return cls(x, y)

    @classmethod
    def read_from_mat(cls, path):
        contents = sio.loadmat(path)
        x = contents['X']
        y = contents['y']
        return cls(x, y)

    def save_to_mat(self, path):
        map_to_args = {'theta': self.final_theta}
        sio.savemat(path, map_to_args)

    def normalize(self):
        """

        :return: normalized features matrix (numpy matrix)
        """
        if self.normalized:
            return self.normalized_features
        else:
            self.normalized_features =\
                np.divide(self.features - self.mean, self.mean,
                          out=np.zeros_like(self.features - self.mean),
                          where=self.mean != 0)
            self.normalized = True
            return self.normalized_features

    def predict(self, new_feature):
        pass

    def train_with_gd(self, alpha=0.01, iterations=10000, plot_cost=False,
                      regularization=False, lbd=0):
        pass

    def train_with_op(self, normalization=False, regularization=False, lbd=0):
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









