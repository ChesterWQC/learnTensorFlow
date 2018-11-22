import scipy.io as sio
import numpy as np
from machine_learning.core.logistic_regression_model import LogisticRegressionTrainingMulti
import random
import matplotlib.pyplot as plt

"""
Linear regression test with custom data
"""
content = sio.loadmat('./data/customData2.mat')

x_train = content['x_train']
y_train = content['y_train']
x_test = content['x_test']
y_test = content['y_test']
training_set = LogisticRegressionTrainingMulti(x_train, y_train)
training_set.train_with_op(regularization=True, lbd=0.1)
count = 0
for i in range(0, 200):
    index = random.randint(0, 499)
    predict = training_set.predict(x_test[index:index+1, :])
    if predict[0, 0] != y_test[index, 0]:
        print 'Gotcha!'
        print y_test[index, 0]
        print predict[0, 0]
        print '-----------------'
        matrix = x_test[index:index+1, :].reshape((20, 20)).T
        plt.imshow(matrix, cmap='gray')
        plt.show()
        count += 1
print '{}%'.format((1 - count / 200.0) * 100)

"""
NN test
"""