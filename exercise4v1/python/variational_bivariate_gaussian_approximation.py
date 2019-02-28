import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal, wishart
from scipy.special import digamma
from mpl_toolkits.mplot3d import Axes3D

## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
  l, V = np.linalg.eigh(Sigma)
  l[l<0] = 0
  t = np.linspace(0.0, 2.0*np.pi, 100)
  xy = np.stack((np.cos(t), np.sin(t)))
  Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
  plt.plot(Txy[:, 0], Txy[:, 1])

## First, we define a bi-variate (2D) Gaussian
mu = np.array([2, 3])
Lambda = np.array([[3, 1], [1, 2]]) # inverse covariance matrix, 2x2
Sigma = np.linalg.pinv(Lambda)

## We want to approximate this bi-variate Gaussian with a factorized
## distribution, i.e. N([x1; x2] | mu, Sigma) \approx p_1(x1) * p2(x2).
## We apply the iterative updates in Eq. 10.12 - 10.15 in Bishops book
## without using the closed-form solution (for illustration).
m1 = np.random.randn()
m2 = np.random.randn()
max_iter = 10
plt.figure(1)
for iteration in range(max_iter):
  plt.clf()
  
  ## First we plot the true Gaussian distribution
  plot_normal(mu, Sigma)
  
  ## Then we plot the current estimate of the factorized distribution
  plot_normal(np.array([m1, m2]), np.diag([1.0/Lambda[0, 0], 1.0/Lambda[1, 1]]))
  #axis([-1, 4, -1, 5]);
  plt.xlim((-1, 4))
  plt.ylim((-1, 5))
  plt.pause(0.5)
  
  ## Now we update the factorized estimate
  m1 = -1 # XXX: do this correctly
  m2 = -1 # XXX: do this correctly

