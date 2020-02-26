from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.datasets import cifar10, mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def rank_data(data, label, flattened_shape, index, plot_type='2d', n_components=30):
    data = data.reshape(-1,flattened_shape)
    standardized_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components).fit(standardized_data)
    data_transformed = pca.transform(standardized_data)
    var_exp = pca.explained_variance_ratio_

    plt.subplot(10,2,2*index)
    x_pos = np.arange(len(var_exp))
    plt.plot(x_pos, np.cumsum(var_exp), '.-')
    plt.ylabel('Cum. var')
    #plt.xticks(rotation=45)

    if plot_type=='2d':
        plt.subplot(10,2,2*index-1)
        plt.scatter(data_transformed[:,0], data_transformed[:,1])
    else:
        ax = fig.add_subplot(10, 2, 2*index-1, projection='3d')
        ax.scatter3D(data_transformed[:,0], data_transformed[:,1], data_transformed[:,2], c=data_transformed[:,2], cmap='Greens')
    
    # Calculate data rank and return a sorted training index ordering based on distance from centroid
    centroid = np.mean(data_transformed[:,0:n_components], axis=0)
    distance = np.linalg.norm(data_transformed[:,0:n_components]-centroid, axis=1)
    sort_index = np.argsort(distance)
    return sort_index

def arrange_data(x_train, y_train, name, flattened_shape, plot_type, n_components):
    ranked_train_list = []
    total = 0
    fig = plt.figure()
    if plot_type=='2d':
        title = 'First 2 Eigen vectors'
    else:
        title = 'First 3 Eigen vectors'

    fig.suptitle(title + ' plot of ' + name + ' dataset with cumulative variance of first ' + str(n_components) + ' Eigen Vectors')

    for i in range(0,10):
        index = np.where(y_train == i)
        train_data = x_train[index]
        print ("The data size for " + name + " label", i ,"is: ", np.size(index))
        # Find out the ordered index of the training data
        ranked_index = rank_data(train_data, np.full((np.size(index)), i), flattened_shape, (i+1), plot_type, n_components)
        
        # Append data based on the ordering
        ranked_train_list.append(train_data[ranked_index])

    return ranked_train_list

def visualize_data(x_train, img_rows, img_cols, channels, dataset_name):
    fig = plt.figure()
    fig.suptitle(dataset_name + ': The Best and the worst samples in the training dataset') 

    for i in range(0,10):
        plt.subplot(10,2,2*(i+1)-1)
        if dataset_name == 'cifar10':
            first = x_train[i][0].reshape((img_rows, img_cols, channels))
            plt.imshow(first, interpolation='nearest')
        else:
            first = x_train[i][0].reshape((img_rows, img_cols))
            plt.imshow(first, cmap='gray')
     
        plt.subplot(10,2,2*(i+1))    
        if dataset_name == 'cifar10':
            last = x_train[i][-1].reshape((img_rows, img_cols, channels))     
            plt.imshow(last, interpolation='nearest')
        else:
            last = x_train[i][-1].reshape((img_rows, img_cols))
            plt.imshow(last, cmap='gray')

# input image dimensions cifar10
img_rows, img_cols, channels = 32, 32, 3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_cifar10 = arrange_data(x_train, y_train.flatten(), 'cifar10', img_rows*img_cols*channels, '2d', 30)
visualize_data(x_train_cifar10, img_rows, img_cols, channels, 'cifar10')

# input image dimensions mnist
img_rows, img_cols, channels = 28, 28, 1
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_mnist = arrange_data(x_train, y_train, 'mnist', img_rows*img_cols*channels, '2d', 90)
visualize_data(x_train_mnist, img_rows, img_cols, channels, 'mnist')


plt.show()
