import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

## Define distribution
mu = (np.array([1, 1]), np.array([-4, 3]))
Sigma = (2*np.eye(2), 4*np.eye(2))
pi_k = [0.3, 0.7]
K = 2

## Plot distribution
X, Y = np.meshgrid(np.linspace(-12.0, 7.0, 100), np.linspace(-6.0, 9.0, 100))
XY = np.stack((X.flatten(), Y.flatten()), axis=1)
Z = np.zeros(len(XY))
for k in range(K):
  Z += pi_k[k] * multivariate_normal.pdf(XY, mean=mu[k], cov=Sigma[k])

plt.contour(X, Y, Z.reshape(X.shape))

## Sample from distribution
N = 500
samples = np.zeros((N, 2))
for n in range(N):
  samples[n] = [0, 0] # XXX: ACTUALLY DRAW A SAMPLE HERE!!!

plt.plot(samples[:, 0], samples[:, 1], '.');

plt.show()

