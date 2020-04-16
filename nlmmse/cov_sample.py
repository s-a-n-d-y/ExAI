import numpy as np
from random import choice
import matplotlib.pyplot as plt
import scipy.stats as ss


def generate_cov(n = 100, a=2, show_cov=False):
    A = np.matrix([np.random.randn(n) + np.random.randn(1)*a for i in range(n)])
    AA_T = A*np.transpose(A)
    C_sqrt = np.diag(np.diag(AA_T)**(-0.5))
    cov = C_sqrt*AA_T*C_sqrt

    if show_cov:
        vals = list(np.array(cov.ravel())[0])
        plt.hist(vals, range=(-1,1))
        plt.show()
        plt.imshow(cov, interpolation=None)
        plt.show()
    return cov


#generate_cov(show_cov=True)

n = 10000
np.random.seed(100)
# Parameters of the mixture components
norm_params = np.array([[5, 1],
                        [1, 1.3],
                        [9, 1.3]])
n_components = norm_params.shape[0]
# Weight of each component, in this case all of them are 1/3
weights = np.ones(n_components, dtype=np.float64) / 3.0
# A stream of indices from which to choose the component
mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
# y is the mixture sample
y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# Theoretical PDF plotting -- generate the x and y plotting positions
xs = np.linspace(y.min(), y.max(), 200)
ys = np.zeros_like(xs)

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

plt.plot(xs, ys)
plt.hist(y, normed=True, bins="fd")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()