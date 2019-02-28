import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal

## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
  l, V = np.linalg.eigh(Sigma)
  l[l<0] = 0
  t = np.linspace(0.0, 2.0*np.pi, 100)
  xy = np.stack((np.cos(t), np.sin(t)))
  Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
  plt.plot(Txy[:, 0], Txy[:, 1])
    
## Load data
data = loadmat('clusterdata2d.mat')['data']
N, D = data.shape

## Initialize parameters
K = 3; # try with different parameters
mu = []
Sigma = []
pi_k = np.ones(K)/K
for _ in range(K):
  # Let mu_k be a random data point:
  mu.append(data[np.random.choice(N)])
  # Let Sigma_k be the identity matrix:
  Sigma.append(np.eye(D))

## Loop until you're happy
max_iter = 100; # XXX: you should find a better convergence check than a max iteration counter
log_likelihood = np.zeros(max_iter)
respons = np.zeros((K, N)) # KxN
for iteration in range(max_iter):
  ## Compute responsibilities
  # XXX: FILL ME IN!
    
  ## Update parameters
  # XXX: FILL ME IN!
  
  ## Compute log-likelihood of data
  log_likelihood[iteration] = -1 # XXX: DO THIS CORRECTLY


## Plot log-likelihood -- did we converge?
plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')

## Plot data
plt.figure(2)
plt.plot(data[:, 0], data[:, 1], '.')
  
for k in range(K):
  plot_normal(mu[k], Sigma[k])
  
plt.show()


