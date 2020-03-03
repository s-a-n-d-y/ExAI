from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
def corr_score(A, B):
    score = []
    if len(A) == len(B):
        for i in range(len(A)): 
            score.append(np.corrcoef(A[i], B[i])[0][1])
    return score

def nn_inference_test_data(x_test, y_test, img_rows, img_cols, channels):
    pass

def score_test_data(x_train, x_test, y_test, img_rows, img_cols, channels, n_components=30):
    score = []
    for i in range(0,10):
        # Find out each train label
        train = x_train[i]
        # Find out the test data and find out the likeness for the corresponding train label
        label_score = []
        for j in range(0,10):
            if i ==j: #Only find how good the test data spread is for its righteous labels
                index = np.where(y_test == j)
                test_data = x_test[index]
                # Append test data to the original list
                data = np.concatenate((train, test_data), axis=0)
                # PCA on train + test data
                data = data.reshape(-1,img_rows*img_cols*channels)
                standardized_data = StandardScaler().fit_transform(data)
                pca = PCA(n_components=n_components).fit(standardized_data)
                data_transformed = pca.transform(standardized_data)
                var_exp = pca.explained_variance_ratio_

                # score based on distance from centroid
                centroid = np.mean(data_transformed[:,0:n_components], axis=0)
                distance = np.linalg.norm(data_transformed[:,0:n_components]-centroid, axis=1)
                
                # Spread of the test data of the same class in similar training data class
                # https://math.stackexchange.com/questions/1525331/calculate-percentage-of-how-far-a-number-is-away-from-a-given-point-in-a-range
                spread = abs(distance.max() - distance.min())
                likelihood = (100*(abs(spread - abs(distance))/spread))[len(train):]
                label_score.append(likelihood)
        score.append(label_score)
    return score

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

def visualize_data(x_train, img_rows, img_cols, channels, dataset_name, n=1):
    fig = plt.figure()
    fig.suptitle(dataset_name + ': The '+str(n)+' best(left) and worst(right) samples in the training dataset') 

    for i in range(0,10):
        for j in range(0,n):
            # Best samples
            ax = plt.subplot(10,2*n,i*10+j+1)
            ax.set_xlabel('Best sample: ' + str(j+1))

            if dataset_name == 'cifar10':
                sample = x_train[i][j].reshape((img_rows, img_cols, channels))
                plt.imshow(sample, interpolation='nearest')
            else:
                sample = x_train[i][j].reshape((img_rows, img_cols))
                plt.imshow(sample, cmap='gray')
        
        for j in range(0,n):
            # Worst samples
            ax = plt.subplot(10,2*n,i*10+j+1+n)
            ax.set_xlabel('Worst sample: ' + str(j+1))
            if dataset_name == 'cifar10':
                sample = x_train[i][-(j+1)].reshape((img_rows, img_cols, channels))     
                plt.imshow(sample, interpolation='nearest')
            else:
                sample = x_train[i][-(j+1)].reshape((img_rows, img_cols))
                plt.imshow(sample, cmap='gray')
    
    # Remove the labels in the inner visualization
    for ax in fig.get_axes():
        ax.label_outer()

# input image dimensions cifar10
img_rows, img_cols, channels = 32, 32, 3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_cifar10 = arrange_data(x_train, y_train.flatten(), 'cifar10', img_rows*img_cols*channels, '2d', 30)
visualize_data(x_train_cifar10, img_rows, img_cols, channels, 'cifar10', n=5)
cifar_test_score = score_test_data(x_train_cifar10, x_test, y_test.flatten(), img_rows, img_cols, channels, 30)
cifar_test_score_nn = nn_inference_test_data(x_test, y_test.flatten(), img_rows, img_cols, channels)
cifar_corr_score = corr_score(cifar_test_score, cifar_test_score)

# input image dimensions mnist
img_rows, img_cols, channels = 28, 28, 1
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_mnist = arrange_data(x_train, y_train, 'mnist', img_rows*img_cols*channels, '2d', 90)
visualize_data(x_train_mnist, img_rows, img_cols, channels, 'mnist', n=5)
mnist_test_score = score_test_data(x_train_mnist, x_test, y_test.flatten(), img_rows, img_cols, channels, 90)
mnist_test_score_nn = nn_inference_test_data(x_test, y_test.flatten(), img_rows, img_cols, channels)
mnist_corr_score = corr_score(mnist_test_score, mnist_test_score)

plt.show()
