import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataGen import DataGen
def main():
    
    # Main method
    np.random.seed(256)

    # 10000 samples, 40 variate feature, 10 mixtures, 3 dimensional data
    data = DataGen(n_samples=10000, n_variate=40, n_mixtures=10, n_dim=2, mixture_weight_random=False, cov_pos_semidefinite=True)
    # print(data.mu)

    # x_gmm = weighted Pdf of the observed samples per mixture
    # x = obs samples (sample x dimension)
    # a = random matrix (Can be thought of input vector, known)
    # s = feature transformation matrix, n_dim x n_sample (Can be thought as weight matrix)
    # w = noise matrix
    x, a, s = data.generate_gmm_samples()
    # Normalize between -1 and 1
    x = x - x.mean()
    x = x / x.max()

    if data.n_dim == 1:
        # 1D plot
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(x, bins=data.n_samples)
        plt.subplot(2, 1, 2)
        for i in range(data.n_mixtures):
            plt.hist(s[i,:,0], bins=data.n_samples)
        plt.show()

    elif data.n_dim == 2:
        #2-D plot
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.scatter(x[:, 0], x[:, 1], marker='o')

        ax = fig.add_subplot(212)
        for i in range(data.n_mixtures):
            ax.scatter(s[i,:,0], s[i,:,1], marker='o', cmap='virdis', s=25, edgecolor='k')

        plt.show()

    elif data.n_dim >=3:
        #3-D plot
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o')
        ax.set_title('Transformed data')

        ax = fig.add_subplot(212, projection='3d')
        for i in range(data.n_mixtures):
            ax.scatter(s[i,:,0], s[i,:,1], s[i,:,2], marker='o', cmap='virdis', s=25, edgecolor='k')
        ax.set_title('Original normalized weighted mixtures of the GMM')

        plt.show()
    else:
        print("No other choice!")


if __name__ == "__main__":
    main()

    '''
    https://colab.research.google.com/drive/1Eb-G95_dd3XJ-0hm2qDqdtqMugLkSYE8#forceEdit=true&sandboxMode=true&scrollTo=DrsHNw9L5fHc
    https://stackoverflow.com/questions/57937185/how-can-i-create-a-random-variables-samples-from-gaussian-mixture-in-python
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html
    '''