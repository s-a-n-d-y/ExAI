import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from mpl_toolkits import mplot3d

# n_centres = No. of mixtures in the GMM
# n_dim = Dimension of the data
def sample_spherical(n_centres, n_dim=3):    
    # uniform points in hypercube
    u_pts = np.random.uniform(low=-1.0, high=1.0, size=(n_dim, n_centres))

    # n dimensional 2 norm squared
    norm2sq = np.sum(u_pts**2, axis=0)

    # mask of points where 2 norm squared  < 1.0
    in_mask = np.less(norm2sq, np.ones(n_centres))
    #in_mask = np.ones(n_centres)

    # use mask to select points, norms inside unit hypersphere
    in_pts = np.compress(in_mask, u_pts, axis=1)
    in_norm2 = np.sqrt(np.compress(in_mask, norm2sq))  # only sqrt selected

    # return normalized points, equivalently, projected to hypersphere surface
    return in_pts/in_norm2


# variate/no. of centers, radius of sphere, dimension
def generate_data(n_centres = 10, radius = 30, dim = 3, cluster_std=1.0):
    c = sample_spherical(n_centres=n_centres, n_dim=dim)
    c = c.T
    corrected_mu = []
    for ele in c:
        corrected_mu.append(tuple(radius*np.array(tuple(ele))))
    X, y_true = make_blobs(n_samples=10000, centers=corrected_mu, random_state=0, cluster_std=cluster_std)
    return X, y_true

def plot_data (dim=3):
    fig = plt.figure()

    if dim ==2:
        X, y_true = generate_data(n_centres = 30, radius = 1, dim = dim, cluster_std=1.0)
        plt.subplot(2, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, s=3, cmap='viridis')

        X, y_true = generate_data(n_centres = 35, radius = 7, dim = dim, cluster_std=1)
        plt.subplot(2, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, s=3, cmap='viridis')

        X, y_true = generate_data(n_centres = 60, radius = 40, dim = dim, cluster_std=1.0)
        plt.subplot(2, 2, 3)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, s=3, cmap='viridis')

        X, y_true = generate_data(n_centres = 100, radius = 100, dim = dim, cluster_std=1.0)
        plt.subplot(2, 2, 4)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, s=3, cmap='viridis')

    if dim == 3:
        ax = fig.add_subplot(221, projection="3d")
        X, y_true = generate_data(n_centres = 30, radius = 1, dim = dim, cluster_std=1.0)
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_true, s=3, cmap='viridis')

        ax = fig.add_subplot(222, projection="3d")
        X, y_true = generate_data(n_centres = 35, radius = 7, dim = dim, cluster_std=1.0)
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_true, s=3, cmap='viridis')

        ax = fig.add_subplot(223, projection="3d")
        X, y_true = generate_data(n_centres = 60, radius = 40, dim = dim, cluster_std=1.0)
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_true, s=3, cmap='viridis')

        ax = fig.add_subplot(224, projection="3d")
        X, y_true = generate_data(n_centres = 100, radius = 100, dim = dim, cluster_std=1.0)
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_true, s=3, cmap='viridis')

    plt.show()

plot_data(2)