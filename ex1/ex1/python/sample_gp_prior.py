## In this exercise, we will draw samples from a Gaussian process prior
## with zero mean and a squared exponential covariance function.
## The function 'kernel' implements the covariance function.
## We will draw samples over the interval [-2, 2].
## The kernel function has parameters lambda and theta (try experimenting with their values)


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

####### HELPER FUNCTIONS #######

# K = kernel(x, y, _lambda, _theta)
#   Evaluate the squared exponential kernel function with parameters
#   lambda and theta.
#
#   x and y should be NxD and MxD matrices. The resulting
#   covariance matrix will be of size NxM.
def kernel(x, y, _lambda, _theta):
  D2 = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean') # pair-wise distances, size: NxM
  K = _theta * np.exp(-0.5 * D2 * _lambda) # NxM
  return K

####### MAIN DEMO SCRIPT #######
lambda0 = 1.0;
theta   = 2.0;

## Define sample interval
M = 100; # dimensionality of samples -- increasing it implies a an increased resultion of the underlying function
xs = np.linspace(-2.0, 2.0, M)

## Evaluate the MxM covariance matrix as k(xs, xs)
Sigma = kernel(xs, xs, lambda0, theta)

## Draw samples
S = 15 # number of samples that we draw
samples = np.random.multivariate_normal(np.zeros(M), Sigma, S) # SxM

## Plot samples
plt.figure(1)
for s in range(S):
  plt.plot(xs, samples[s])
plt.title('Samples from GP')
#plt.show()

## Plot samples from a normal distribution with zero mean and identity covariance
samples_iid = np.random.multivariate_normal(np.zeros(M), np.eye(M), S) # SxM
plt.figure(2)
for s in range(S):
  plt.plot(xs, samples_iid[s])
plt.title('Samples from isotropic normal distribution')
#plt.show()

## Show covariance matrix
plt.figure(3)
plt.imshow(Sigma, extent=[-2, 2, -2, 2])
plt.colorbar()
plt.title('Covariance matrix')

plt.show()

