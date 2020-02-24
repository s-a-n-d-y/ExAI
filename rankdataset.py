from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential


def arrange_data(x_train, y_train, name=''):
    train = []
    for i in range(0,10):
        index = np.where(y_train == i)
        data = x_train[index]
        print ("The data size for " + name + " label", i ,"is: ", np.size(index))
        train.append(data)
    return train


# input image dimensions cifar10
img_rows, img_cols = 32, 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_cifar10 = arrange_data(x_train, y_train.flatten(), 'cifar10')

# input image dimensions mnist
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_mnist = arrange_data(x_train, y_train, 'mnist')

