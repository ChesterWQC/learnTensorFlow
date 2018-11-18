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


def normalize_feature(matrix):
    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0)
    normalized = (matrix - mu) / sigma
    return [normalized, mu, sigma]


def cost_func(x, y, theta, lbd=0):
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
    j = np.sum(np.square((x.dot(theta) - y))) / 2 / m
    if lbd != 0:
        j_r = np.sum(theta[1:, 0:1]) * lbd / 2 / m
        j = j + j_r
    return j


def cost_flatten(theta, x, y, lbd=0):
    theta = theta.reshape((theta.size, 1))
    return cost_func(x, y, theta, lbd)


# def cost_reg(x, y, theta, lbd):
#     # J_2 = lambda/2/m * ones(1,len_theta-1) * (theta(2:len_theta,1) .^ 2);
#     m = x.shape[0]
#     j_1 = cost(x, y, theta)
#     j_2 = np.sum(theta[1:, 0:1]) * lbd / 2 / m
#     return j_1 + j_2


def grad(x, y, theta, lbd=0):
    """

    :param x: numpy matrix of features m * n+1
    :param y: numpy column vector of result m * 1
    :param theta: numpy column vector of params n+1 * 1
    :param lbd: lambda value for regularization
    :return: gradient of current theta n+1 * 1
    """
    m = x.shape[0]
    delta = ((x.dot(theta) - y).T.dot(x)).T / m
    if lbd != 0:
        delta_r = np.vstack((0, theta[1:, 0:1] * lbd / m))
        delta = delta + delta_r
    return delta


# def grad_reg(x, y, theta, lbd):
#     # grad = 1/m * (((hX - y)' * X)') + [0;lambda/m*theta(2:len_theta,1)];
#     m = x.shape[0]
#     delta_1 = grad(x, y, theta)
#     delta_2 = np.vstack((0, theta[1:, 0:1] * lbd / m))
#     return delta_1 + delta_2


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
