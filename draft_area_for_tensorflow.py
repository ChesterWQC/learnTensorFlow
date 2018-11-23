import tensorflow as tf
from tensorflow.python.keras import layers
import scipy.io as sio
import numpy as np
import datetime


contents = sio.loadmat('./data/customData2')
x_train = contents['x_train']
y_train = contents['y_train']
x_test = contents['x_test']
y_test = contents['y_test']

print '{},{},{},{}'.format(x_train.shape, y_train.shape,
                           x_test.shape, y_test.shape)
y_train_processed = np.zeros((y_train.shape[0], 10))
y_test_processed = np.zeros((y_test.shape[0], 10))
for i in range(0, 10):
    y_train_processed[:, i:i+1] = (y_train == i).astype(int)
    y_test_processed[:, i:i+1] = (y_test == i).astype(int)
#
#
# model = tf.keras.Sequential()
# layer_1 = layers.Dense(64, activation='sigmoid', kernel_initializer='orthogonal',
#                        kernel_regularizer=tf.keras.regularizers.l1(0.01))
# layer_2 = layers.Dense(64, activation='relu', kernel_initializer='orthogonal',
#                        kernel_regularizer=tf.keras.regularizers.l1(0.01))
# layer_3 = layers.Dense(10, activation='softmax', kernel_initializer='orthogonal',
#                        kernel_regularizer=tf.keras.regularizers.l1(0.01))
# model.add(layer_1)
# model.add(layer_2)
# model.add(layer_3)
# date1 = datetime.datetime.now()
# model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.categorical_crossentropy,
#               metrics=[tf.keras.metrics.categorical_accuracy])
# model.fit(x_train, y_train_processed, epochs=1000, batch_size=500, validation_data=(x_test, y_test_processed))
# date2 = datetime.datetime.now()
# print date1
# print date2
# result = model.predict(x_test, batch_size=32)
# sio.savemat('./data/tfResult.mat', {
#     'result': result
# })

predict = sio.loadmat('./data/tfResult.mat')['result']
print predict.shape
print y_test_processed[0,:]

