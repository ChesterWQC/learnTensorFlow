"""

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def plot_cost(cost):
    plt.plot(cost)
    plt.xlabel("Iterations")
    plt.ylabel("J")
    plt.show()


def sigmoid(z):
    return (1 + np.exp(0 - z)) ** -1


def normalize_feature(matrix):
    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0)
    normalized = (matrix - mu) / sigma
    return [normalized, mu, sigma]


def cost_func(x, y, theta, lbd=0):
    # J_1 = 1/m * ((-y)' * log(hX) - (1 - y)' * log(1 - hX));
    """

    :param x: numpy matrix of features m * n+1
    :param y: numpy column vector of result m * 1
    :param theta: numpy column vector of params n+1 * 1
    :param lbd: lambda value for regularization
    :return: cost for current theta n+1 * 1
    """
    # theta is (n + 1, 1)
    # x is (m, n + 1)
    # y is (m, 1)
    m = x.shape[0]
    # print x.shape
    # print theta.shape
    # print x.dot(theta).shape
    h_x = sigmoid(x.dot(theta))
    j = np.sum(((-y).T.dot(np.log(h_x)) - (1 - y).T.dot(np.log(1 - h_x)))) / m
    if lbd != 0:
        j_r = np.sum(theta[1:, 0:1]) * lbd / 2 / m
        j = j + j_r
    return j


def cost_flatten(theta, x, y, lbd=0):
    theta = theta.reshape((theta.size, 1))
    return cost_func(x, y, theta, lbd)


def grad(x, y, theta, lbd=0):
    """

    :param x: numpy matrix of features m * n+1
    :param y: numpy column vector of result m * 1
    :param theta: numpy column vector of params n+1 * 1
    :param lbd: lambda value for regularization
    :return: gradient of current theta n+1 * 1
    """
    m = x.shape[0]
    h_x = sigmoid(x.dot(theta))
    delta = ((h_x - y).T.dot(x)).T / m
    if lbd != 0:
        delta_r = np.vstack((0, theta[1:, 0:1] * lbd / m))
        delta = delta + delta_r
    return delta


def grad_flatten(theta, x, y, lbd=0):
    theta = theta.reshape((theta.size, 1))
    return grad(x, y, theta, lbd)


def gradient_descent(x, y, theta, alpha=0.01, iterations=10000,
                     display_cost=False, regularization=False, lbd=0):
    """

    :param x: numpy matrix of features m * n+1
    :param y: numpy column vector of result m * 1
    :param theta: numpy column vector of params n+1 * 1
    :param alpha: descent factor
    :param iterations: iteration times
    :param display_cost: display the current cost on each iteration
    :param regularization: choose to regularize to avoid over-fitting
    :param lbd: lambda value for regularization
    :return: trained theta n+1 * 1
    """

    cost_history = []
    if regularization:
        for i in range(iterations):
            delta = grad(x, y, theta, lbd)
            theta = theta - alpha * delta
            cur_cost = cost_func(x, y, theta, lbd)
            cost_history.append(cur_cost)
            if display_cost:
                print 'The current cost is {} in iteration {}'\
                    .format(cur_cost, i)
    else:
        for i in range(iterations):
            delta = grad(x, y, theta)
            theta = theta - alpha * delta
            cur_cost = cost_func(x, y, theta)
            cost_history.append(cur_cost)
            if display_cost:
                print 'The current cost is {} in iteration {}'\
                    .format(cur_cost, i)

    return theta, cost_history


def normal_equation(x, y):
    """

    :param x: numpy matrix of features m * n+1
    :param y: numpy column vector of result m * 1
    :return: trained theta n+1 * 1
    """
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta


def optimize_function(x, y, theta, regularization=False, lbd=0):
    n = x.shape[1]
    theta = theta.reshape((n,))
    if regularization:
        result = op.minimize(fun=cost_flatten, x0=theta, args=(x, y, lbd),
                             method='TNC', jac=grad_flatten)
    else:
        result = op.minimize(fun=cost_flatten, x0=theta, args=(x, y),
                             method='TNC', jac=grad_flatten)

    return result.x
