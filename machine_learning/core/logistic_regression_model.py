"""

"""
import scipy.io as sio
import numpy as np
import machine_learning.util.logistic_regression as lor
from machine_learning.core.base_model import RegressionTrainingSet


class LogisticRegressionTraining(RegressionTrainingSet):

    def __init__(self, data=None):
        super(LogisticRegressionTraining, self).__init__(data)

    def predict_binary_classification(self, new_feature):
        """

        :param new_feature: m * n matrix
        :return: m * 1 matrix of result
        """
        if not self.trained:
            print 'Please train the model first!'
            return
        if self.normalized:
            new_feature = (new_feature - self.mean) / self.sigma
        feature = np.hstack((np.ones((new_feature.shape[0], 1)), new_feature))
        result = feature.dot(self.final_theta)
        for i in range(new_feature.shape[0]):
            if result[i, 0] >= 0.5:
                result[i, 0] = 1
            else:
                result[i, 0] = 0
        return result

    def train_with_gd(self, alpha=0.01, iterations=10000, plot_cost=False,
                      normalization=True, regularization=False, lbd=0):
        if self.trained:
            print 'This model has already been trained!'
        else:
            if normalization:
                features = self.normalize()
            else:
                features = self.features
            processed_features = np.hstack((np.ones((self.sample_size, 1)),
                                            features))
            self.final_theta,\
                self.cost_history = lor.gradient_descent(
                    processed_features, self.results, self.init_theta, alpha,
                    iterations, regularization=regularization, lbd=lbd)
            self.trained = True
        if plot_cost:
            self.plot_cost()

    def train_with_op(self, normalization=True, regularization=False, lbd=0):
        if self.trained:
            print 'This model has already been trained'
        else:
            if normalization:
                features = self.normalize()
            else:
                features = self.features
            processed_features = np.hstack((np.ones((self.sample_size, 1)),
                                            features))
            self.final_theta = lor.optimize_function(
                processed_features, self.results, self.init_theta,
                regularization=regularization, lbd=lbd)
            self.trained = True

