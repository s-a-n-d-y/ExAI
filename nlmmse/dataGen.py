import numpy as np
from scipy.linalg import block_diag
from sklearn.datasets import make_blobs

# A, known random matrix of dim qxp
# S, Gaussian samples of dim px1, dim of features = variate
# W, Gaussian additive noise of dim qx1
class DataGen:   
    def __init__(self, n_samples=1000, n_variate=3, n_mixtures=10, n_dim=2, mixture_weight_random=True, cov_pos_semidefinite=True): 
        # No. of samples per mixture in the GMM    
        self.n_mixtures = n_mixtures 
        # The feature distributions can have diff. variates
        # Right now all variates are same
        self.n_variate = n_variate
        self.n_dim = n_dim
        
        # GMM parameters
        self.cov_pos_semidefinite = cov_pos_semidefinite

        # mu = no. of mixtures x features, Considering univariate mu
        self.mu = np.random.randint(low = 0, high = 50, size=(self.n_mixtures, self.n_variate))
        
        # Assuming random std. dev within clusters
        self.cov = np.empty(shape=[self.n_mixtures, self.n_variate, self.n_variate])
        for i in range(self.n_mixtures):
            self.cov[i,:,:] = self.generate_cov(dim=self.n_variate, display_cov = False, cov_pos_semidefinite = self.cov_pos_semidefinite)

        # Total no of samples
        self.n_samples = n_samples

        if mixture_weight_random:
            # Generate random weights for the GMM from Dirichlet distribution
            self.sample_len = (int)(np.random.dirichlet(np.ones(self.n_mixtures),size=1)*self.n_samples)
        else:  
            self.sample_len = (int)(self.n_samples/self.n_mixtures)*np.ones((1, self.n_mixtures), dtype=np.int16)

        # Assumption: In all mixtures the noise profile is same
        # To change that add one corresponding dimension for the mixture dimension
        # self.noise_sigma = np.ones(shape=[self.n_mixtures, self.n_dim, 1])
        # Noise parameters
        self.noise_mu = np.zeros(shape=[self.n_dim, 1])
        self.noise_sigma = np.ones(shape=[self.n_dim, 1])
    
    def generate_gmm_samples(self):
        S, Y = make_blobs(
                n_samples=self.sample_len, 
                n_features=self.n_dim, centers=self.mu, 
                cluster_std=1.0, 
                center_box=(-10.0, 10.0), 
                shuffle=True, 
                random_state=None
                )

        A = np.random.normal(0, 1, (self.n_dim, self.n_variate))
        W = np.zeros(shape=[self.n_samples, self.n_dim])
        X = A*S.T + W 
        return X, A, S

    '''
    # Returns randomly generated samples X
    def generate_gmm_samples(self):
        A = np.random.normal(0, 1, (self.n_dim, self.n_variate))      
        S = np.zeros(shape=[self.n_mixtures, self.n_samples, self.n_variate])
        X = np.zeros(shape=[self.n_samples, self.n_dim])
        W = np.zeros(shape=[self.n_samples, self.n_dim])


        for i in range(self.n_dim):
            # W = error matrix, 0 mean std. dev = 1
            W[:,i] = np.random.normal(0, 1, self.n_samples)

        # Multiply each mixture by alpha, where sum(alpha) = 1, generated from Dirichlet
        for i in range(self.n_mixtures):
            if i == 0:
                S[i,0:self.sample_len[0,i],:] = np.random.multivariate_normal(self.mu[i,:], self.cov[i,:,:], self.sample_len[0,i])
            else:
                S[i,self.sample_len[0,i-1]:self.sample_len[0,i],:] = np.random.multivariate_normal(self.mu[i,:], self.cov[i,:,:], self.sample_len[0,i])
                X_GMM[i,:,:] = np.dot(S[i,self.sample_len[0,i-1]:self.sample_len[0,i],:], A.T) + W
        
        # Observed samples, transformation (NN), original feature distribution, noise
        return X_GMM, A, S, W
    '''

    # Full rank random correlation matrix
    def generate_cov(self, dim, a=2, display_cov=False, cov_pos_semidefinite=True):
        if cov_pos_semidefinite: 
            # Normalized cov
            # https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor/124554
            # 'a' is a tuning parameter
            A = np.matrix([np.random.randn(dim) + np.random.randn(1)*a for i in range(dim)])
            # A = np.matrix([np.random.random(dim) + np.random.random(1)*a for i in range(dim)])
            AA_T = A*np.transpose(A)
            C_sqrt = np.diag(np.diag(AA_T)**(-0.5))
            cov = C_sqrt*AA_T*C_sqrt
        else:
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
    
    # Returns estimate of the original feature distribution, S
    def estimate_feature_nlmmse(self):  
        for i in range(self.n_mixtures):
            mu = np.append(self.mu[i,:], self.noise_mu)
            cov = block_diag(self.cov[i,:,:], 1)
        
        #######################################################
        S = np.empty(shape=[self.n_samples, self.n_projection])
        Y = np.empty(shape=self.n_samples)
    
        # W = error matrix, 0 mean std. dev = 1
        W = np.random.normal(0, 1, self.n_projection)
        A = np.random.normal(0, 1, (self.n_projection, self.n_variate))

        # Multiply each mixture by alpha, where sum(alpha) = 1, generated from Dirichlet
        for i in range(self.n_mixtures):
            mu = np.append(self.mu[i,:], 0)
            cov = block_diag(self.cov[i,:,:], 1)
            S[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] =  self.alpha[:,i].item()*np.random.multivariate_normal(mu, cov, self.n_samples_per_mix)
            Y[i*self.n_samples_per_mix:(i+1)*self.n_samples_per_mix] = i

        X = np.dot(A, S) + W 
        
        # Observed samples, original feature distribution, known labels
        return S