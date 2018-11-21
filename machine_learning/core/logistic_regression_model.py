"""

"""
import numpy as np
import machine_learning.util.logistic_regression as lor
from machine_learning.core.base_model import RegressionTrainingSet, TrainingType


class LogisticRegressionTrainingBinary(RegressionTrainingSet):

    def __init__(self, x, y):
        super(LogisticRegressionTrainingBinary, self).__init__(x, y)
        self._type = TrainingType.LOGISTIC_REGRESSION_BINARY

    def predict(self, new_feature):
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
        result = feature.dot(self.final_theta.T)
        for i in range(new_feature.shape[0]):
            if result[i, 0] >= 0.5:
                result[i, 0] = 1
            else:
                result[i, 0] = 0
        return result

    def train_with_gd(self, alpha=0.01, iterations=10000, plot_cost=False,
                      normalization=False, regularization=False, lbd=0):
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

    def train_with_op(self, normalization=False, regularization=False, lbd=0):
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


class LogisticRegressionTrainingMulti(RegressionTrainingSet):

    def __init__(self, x, y):
        super(LogisticRegressionTrainingMulti, self).__init__(x, y)
        self._type = TrainingType.LOGISTIC_REGRESSION_MULTI
        self._labels_set = set()
        for i in y.reshape((self.sample_size,)):
            self._labels_set.add(i)
        self.label_number = len(self._labels_set)
        self.labels = list(self._labels_set)
        self.init_theta = np.zeros((
            self.label_number, self.feature_number + 1)) # labels * n+1 matrix
        self.final_theta = self.init_theta

    def predict(self, new_feature):
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
        result = feature.dot(self.final_theta.T)
        indices = np.argmax(result, axis=1)  # shape (m,)
        predict = np.array([self.labels[i] for i in indices])  # shape (m,)
        return predict.reshape(predict.shape[0], 1)

    def train_with_gd(self, alpha=0.01, iterations=10000, plot_cost=False,
                      normalization=False, regularization=False, lbd=0):
        if self.trained:
            print 'This model has already been trained!'
        else:
            if normalization:
                features = self.normalize()
            else:
                features = self.features
            processed_features = np.hstack((np.ones((self.sample_size, 1)),
                                            features))
            label_index = 0
            for label in self.labels:
                init_theta_single = self.init_theta[
                                    label_index:label_index + 1, :]  # 1 * n + 1
                final_theta_single,\
                    cost_history_single = lor.gradient_descent(
                        processed_features, (self.results == label),
                        init_theta_single, alpha, iterations,
                        regularization=regularization, lbd=lbd)
                self.final_theta[label_index:label_index + 1, :]\
                    = final_theta_single
                self.cost_history.append(cost_history_single)
                label_index = label_index + 1
            self.trained = True
        if plot_cost:
            self.plot_cost()

    def train_with_op(self, normalization=False, regularization=False, lbd=0):
        if self.trained:
            print 'This model has already been trained'
        else:
            if normalization:
                features = self.normalize()
            else:
                features = self.features
            processed_features = np.hstack((np.ones((self.sample_size, 1)),
                                            features))
            label_index = 0
            for label in self.labels:
                init_theta_single = self.init_theta[
                                    label_index:label_index + 1, :]  # 1 * n + 1
                final_theta_single = lor.optimize_function(
                    processed_features, (self.results == label),
                    init_theta_single, regularization=regularization, lbd=lbd)
                self.final_theta[label_index:label_index + 1, :] \
                    = final_theta_single
                label_index = label_index + 1
            self.trained = True

    def train_with_op_with_gray_scale_processing(self, normalization=False,
                                                 regularization=False, lbd=0):
        if self.trained:
            print 'This model has already been trained'
        else:
            if normalization:
                features = self.normalize()
            else:
                max_val = np.max(self.features, axis=1).\
                              reshape((self.features.shape[0], 1)) * 1.0

                features = np.divide(np.abs(self.features), max_val,
                                     out=np.zeros_like(self.features),
                                     where=max_val != 0)
            processed_features = np.hstack((np.ones((self.sample_size, 1)),
                                            features))
            label_index = 0
            for label in self.labels:
                init_theta_single = self.init_theta[
                                    label_index:label_index + 1, :]  # 1 * n + 1
                final_theta_single = lor.optimize_function(
                    processed_features, (self.results == label),
                    init_theta_single, regularization=regularization, lbd=lbd)
                self.final_theta[label_index:label_index + 1, :] \
                    = final_theta_single
                label_index = label_index + 1
            self.trained = True

