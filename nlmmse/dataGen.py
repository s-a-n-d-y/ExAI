import numpy as np, numpy.random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag

# A, known random matrix of dim qxp
# S, Gaussian samples of dim px1
# W, Gaussian additive noise of dim qx1
class DataGen:   
    # n_samples = p
    # n_features = q
    def __init__(self, n_samples_per_mix, n_variate=3, n_mixtures=10): 
        # No. of samples per mixture in the GMM    
        self.n_samples_per_mix = n_samples_per_mix
        self.n_mixtures = n_mixtures 
        # The feature distributions can have diff. variates
        # Right now all variates are same
        self.n_variate = n_variate
        
        # GMM parameters
        # mu = no. of mixtures x features, Considering univariate mu
        self.mu = np.random.randint(low = 0, high = 50, size=(self.n_mixtures, self.n_variate))
        
        # Assuming random std. dev within clusters
        self.cov = np.empty(shape=[self.n_mixtures, self.n_variate, self.n_variate])
        for i in range(self.n_mixtures):
            self.cov[i,:,:] = self.generate_cov(dim=self.n_variate, display_cov = False)

        # Generate random weights for the GMM from Dirichlet distribution
        self.alpha = np.random.dirichlet(np.ones(self.n_mixtures),size=1)

        # Total no of samples
        self.n_samples = n_samples_per_mix*n_mixtures

    '''
    # Returns randomly generated samples X
    def generate_gmm_samples(self):           
        S = np.empty(shape=[self.n_samples, self.n_variate])
        Y = np.empty(shape=self.n_samples)
    
        # W = error matrix, 0 mean std. dev = 1
        W = np.random.normal(0, 1, self.n_samples_per_mix)
        A = np.random.normal(0, 1, (self.n_samples, self.n_variate))

        # Multiply each mixture by alpha, where sum(alpha) = 1, generated from Dirichlet
        for i in range(self.n_mixtures):
            mu = np.append(self.mu[i,:], 0)
            cov = block_diag(self.cov[i,:,:], 1)
            S[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] =  self.alpha[:,i].item()*np.random.multivariate_normal(mu, cov, self.n_samples_per_mix)
            Y[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] = i

        X = np.dot(A, S) + W 
        
        # Observed samples, original feature distribution, known labels
        return X, S, Y
    '''
    # Returns randomly generated samples X
    def generate_gmm_samples(self):           
        S = np.zeros(shape=[self.n_samples, self.n_variate])
        Y = np.empty(shape=self.n_samples)
    
        # Multiply each mixture by alpha, where sum(alpha) = 1, generated from Dirichlet
        for i in range(self.n_mixtures):
            S[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] =  self.alpha[:,i].item()*np.random.multivariate_normal(self.mu[i,:], self.cov[i,:,:], self.n_samples_per_mix)
            Y[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] = i

        W = np.random.normal(0, 1, self.n_samples)
        A = np.random.normal(0, 1, (self.n_samples, self.n_variate))

        # X = np.dot(A, S) + W 
        # Observed samples, an assumption of feature distribution, known labels
        return S, Y #X

    # Full rank random correlation matrix
    def generate_cov(self, dim, a=2, display_cov=False):
        '''
        # Normalized cov
        # https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor/124554
        # 'a' is a tuning parameter
        A = np.matrix([np.random.randn(dim) + np.random.randn(1)*a for i in range(dim)])
        # A = np.matrix([np.random.random(dim) + np.random.random(1)*a for i in range(dim)])
        AA_T = A*np.transpose(A)
        C_sqrt = np.diag(np.diag(AA_T)**(-0.5))
        cov = C_sqrt*AA_T*C_sqrt
        '''
        # A simplier cov. May not be normalized!
        # The data looks nicer
        cov = np.random.rand(dim, dim)
        cov = cov - np.diag(np.diag(cov)) + np.diag(np.random.randint(low = 1, high = 10, size=(dim)))
        
        if display_cov:
            #vals = list(np.array(cov.ravel())[0])
            #plt.hist(vals, range=(-1,1))
            #plt.show()
            plt.imshow(cov, interpolation=None)
            plt.show()

        return cov


# Main method
np.random.seed(256)

# 1000 samples, 3 variate, 5 mixtures
data = DataGen(500, 3, 5)
y, x = data.generate_gmm_samples()
print(data.mu)

# 3D plot

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[:, 0], y[:, 1], y[:, 2], marker='o', c=x, s=25, edgecolor='k')
plt.show()


# 1D plot
'''
plt.figure()
plt.hist(y,  bins=1000)
'''

#2-D plot
'''
plt.figure()
plt.scatter(y[:, 0], y[:, 1], marker='o', c=x, s=25, edgecolor='k')
plt.show()
'''



