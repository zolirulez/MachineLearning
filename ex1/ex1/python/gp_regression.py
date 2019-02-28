import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
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

## Load data
## We subsample the data, which gives us N pairs of (x, y)
M = 1000
data = loadmat('weather.mat')
x = np.arange(0, M, 20)
y = data['TMPMAX'][x]
N = len(y);

## Standardize data to have zero mean and unit variance
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## We want to predict values at x_* (denoted xs in the code)
xs = np.linspace(np.min(x), np.max(x), M)
#xs = np.linspace(-2, 2, M) # Mx1 --- try predicting over this interval instead

## Initial kernel parameters -- once you have a GP regressor,
## you should play with these parameters and see what happens
lambda0 = 100.0
theta  = 2.0

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1.0)**2

## Compute covariance (aka "kernel") matrices
## XXX: FILL ME IN!
K   = np.zeros((N, N)) # NxN
Ks  = np.zeros((N, M)) # NxM
Kss = np.zeros((M, M)) # MxM
 
## Compute conditional mean p(y_* | x, y, x_*)
## XXX: FILL ME IN!
mu = np.zeros(M) # Mx1
Sigma = np.zeros((M, M)) # MxM

## Plot the mean prediction
plt.figure(1)
plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
plt.plot(xs, mu) # mean prediction
plt.title('Mean prediction')

## Plot samples
plt.figure(2)
plt.plot(x, y, 'ko', markerfacecolor='k') # raw data
S = 50 # number of samples
samples = np.random.multivariate_normal(mu.reshape(-1), Sigma, S) # SxM
for s in range(S):
  plt.plot(xs, samples[s])
plt.title('Samples')

## Evaluate log-likelihood for a range of lambda's
#Q = 100
#possible_lambdas = np.linspace(1, 300, Q)
#loglikelihood = np.zeros(Q)
#for k in range(Q):
#  lambda_k = possible_lambdas[k]
#  # Compute log-likelihood of data for lambda_k
#  # XXX: FILL ME IN!
#
#idx = np.argmax(loglikelihood)
#lambda_opt = possible_lambdas[idx]
#plt.figure(3)
#plt.plot(possible_lambdas, loglikelihood)
#plt.plot(possible_lambdas[idx], loglikelihood[idx], '*')
#plt.title('Log-likelihood for \lambda');
#plt.xlabel('\lambda')
#plt.show()

plt.show()

