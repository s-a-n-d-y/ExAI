from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ModelSelector:

    def __init__(self, img_rows, img_cols, channels, n_components, dataset_name, plot_type, n_plot_sample, n_classes=10): 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.dataset_name = dataset_name
        self.n_components = n_components
        self.plot_type = plot_type
        self.n_plot_sample = n_plot_sample
        self.n_classes = n_classes
        self.flattened_shape = self.img_rows*self.img_cols*self.channels
    
    # https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    # Returns a corrleation score in between 2D lists
    # List A structure: [[label 0 rank 1, label 0 rank 2, .... label 0 rank N],
    #                    [label 1 rank 1, label 1 rank 2, .... label 1 rank N],
    #                    [label 2 rank 1, label 2 rank 2, .... label 2 rank N],
    #                           .               .                    .    
    #                           .               .                    .
    #                    [label N rank 1, label N rank 2, .... label N rank N]]
    def corr_score_per_label(self, A, B):    
        score = []
        if len(A) == len(B):
            for i in range(len(A)): 
                norm1 = A[i]/np.sqrt(np.dot(A[i], A[i]))
                norm2 = B[i]/np.sqrt(np.dot(B[i], B[i]))
                d = norm1 - norm2 
                score.append(d/np.sqrt(np.dot(d, d)))
        return score
        
        '''
        A_mA = A - A.mean(1)[:,None]
        B_mB = B - B.mean(1)[:,None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
        '''

    def nn_score_test_data(self, model, x_test_ranked):
        predictions = []
        for i in range(0,self.n_classes):
            test_data = x_test_ranked[i]
            print ("The test data size for " + self.dataset_name + " label", i ,"is: ", len(test_data))
            # Reshape it to adjust to NN input predictor
            test_data = test_data.reshape(test_data.shape[0], self.img_rows, self.img_cols, self.channels)
            #test_data = test_data.astype('float64')
            test_data /= 255.0
            predictions.append(model.predict(test_data)[:,i])        
        return predictions

    def score_test_data(self, x_train_ranked, x_test, y_test):
        score = []
        test_ranked = []
        for i in range(0,self.n_classes):
            # Find out each ranked train label
            train_ranked = x_train_ranked[i]
            # Find out the test data and find out the likeness for the corresponding ranked train label
            for j in range(0,self.n_classes):
                if i ==j: #Only find how good the test data spread is for its righteous labels
                    index = np.where(y_test == j)
                    test_data = x_test[index]
                    # Append test data to the original list
                    data = np.concatenate((train_ranked, test_data), axis=0)
                    # Find out the ranked train + test data
                    _, data_transformed, norm_spread  = self.rank_data(data, (i+1))    
                    # Finally only calculate the spread of the test data i.e. why I chose from index after ranked train data
                    norm_spread = norm_spread[len(train_ranked):]
                    # Spread less from centroid means good model i.e. why sorted with a negative spread
                    sort_index = np.argsort(-norm_spread)
                    test_ranked.append(data[sort_index])
                    # Congugate of spread should be the score to compare with NN classification
                    score.append(1-norm_spread[sort_index])
        return score, test_ranked

    def rank_data(self, data, index, show_plot=False):
        data = data.reshape(-1,self.flattened_shape)
        standardized_data = StandardScaler().fit_transform(data)
        pca = PCA(n_components=self.n_components).fit(standardized_data)
        data_transformed = pca.transform(standardized_data)
        var_exp = pca.explained_variance_ratio_

        if show_plot:
            plt.subplot(self.n_classes,2,2*index)
            x_pos = np.arange(len(var_exp))
            plt.plot(x_pos, np.cumsum(var_exp), '.-')
            plt.ylabel('Cum. var')
            #plt.xticks(rotation=45)

            if self.plot_type=='2d':
                plt.subplot(self.n_classes,2,2*index-1)
                plt.scatter(data_transformed[:,0], data_transformed[:,1])
            else:
                ax = fig.add_subplot(self.n_classes, 2, 2*index-1, projection='3d')
                ax.scatter3D(data_transformed[:,0], data_transformed[:,1], data_transformed[:,2], c=data_transformed[:,2], cmap='Greens')
        
        # Calculate data rank and return a sorted training index ordering based on spread from centroid
        centroid = np.mean(data_transformed, axis=0)
        # Then find the spread of the all data of the same class in similar ranked training data class
        # https://math.stackexchange.com/questions/1525331/calculate-percentage-of-how-far-a-number-is-away-from-a-given-point-in-a-range
        spread = np.linalg.norm(data_transformed-centroid[:,None].T, axis=1)
        norm_spread = (abs(spread/abs(spread.max() - spread.min())))
        sort_index = np.argsort(norm_spread)
        return sort_index, data_transformed, norm_spread

    def arrange_data(self, x_train, y_train):
        ranked_train_list = []
        total = 0
        fig = plt.figure()
        if self.plot_type=='2d':
            title = 'First 2 Eigen vectors'
        else:
            title = 'First 3 Eigen vectors'

        fig.suptitle(title + ' plot of ' + self.dataset_name + ' dataset with cumulative variance of first ' + str(self.n_components) + ' Eigen Vectors')

        for i in range(0,self.n_classes):
            index = np.where(y_train == i)
            train_data = x_train[index]
            print ("The training data size for " + self.dataset_name + " label", i ,"is: ", np.size(index))
            # Find out the ordered index of the training data
            ranked_index, _, _  = self.rank_data(train_data, (i+1), show_plot=True)
            # Append data based on the ordering
            ranked_train_list.append(train_data[ranked_index])
        return ranked_train_list

    def visualize_data(self, x_data, data_type='train'):
        fig = plt.figure()
        fig.suptitle(self.dataset_name + ': The '+str(self.n_plot_sample)+' best(left) and worst(right) samples in the '+ data_type +'ing dataset') 

        for i in range(0,self.n_classes):
            for j in range(0,self.n_plot_sample):
                # Best samples
                ax = plt.subplot(self.n_classes,2*self.n_plot_sample,i*self.n_classes+j+1)
                ax.set_xlabel('Best sample: ' + str(j+1))

                if self.dataset_name == 'cifar10':
                    sample = x_data[i][j].reshape((self.img_rows, self.img_cols, self.channels))
                    sample = sample/255
                    plt.imshow(sample, interpolation='nearest')
                else:
                    sample = x_data[i][j].reshape((self.img_rows, self.img_cols))
                    plt.imshow(sample, cmap='gray')
            
            for j in range(0,self.n_plot_sample):
                # Worst samples
                ax = plt.subplot(self.n_classes,2*self.n_plot_sample,i*self.n_classes+j+1+self.n_plot_sample)
                ax.set_xlabel('Worst sample: ' + str(j+1))
                if self.dataset_name == 'cifar10':
                    sample = x_data[i][-(j+1)].reshape((self.img_rows, self.img_cols, self.channels)) 
                    sample = sample/255   
                    plt.imshow(sample, interpolation='nearest')
                else:
                    sample = x_data[i][-(j+1)].reshape((self.img_rows, self.img_cols))
                    plt.imshow(sample, cmap='gray')
        
        # Remove the labels in the inner visualization
        for ax in fig.get_axes():
            ax.label_outer()

    def visualize_correlated_inference(self, A, B, label_A, label_B):
        fig = plt.figure()
        
        for i in range(len(A)):
            list_A_i = A[i]
            list_B_i = B[i]
            plt.subplot(10,2,2*i+1)

            plt.plot(range(0, len(list_A_i)), list_A_i.tolist(), color='blue', alpha=0.5, label=label_A)
            plt.plot(range(0, len(list_A_i)), list_B_i.tolist(), color='green', alpha=0.5, label=label_B)
            plt.legend(loc="lower right")

            plt.subplot(self.n_classes,2,2*i+2)

            my_dict = {'A': list_A_i.tolist(), 'B': list_B_i.tolist()}
            plt.boxplot(my_dict.values(), labels=(label_A,label_B), showfliers=False)

        # Remove the labels in the inner visualization
        for ax in fig.get_axes():
            ax.label_outer()
