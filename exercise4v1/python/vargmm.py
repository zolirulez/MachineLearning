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

def logdet(Sigma):
  (s, ulogdet) = np.linalg.slogdet(Sigma)
  return s*ulogdet

## Load data
data = loadmat('clusterdata2d.mat')['data']
N, D = data.shape

## Number of components/clusters
K = 10

## Priors
alpha0 = 1e-3 # Mixing prior (small number: let the data speak)
m0 = np.zeros(D); beta0 = 1e-3 # Gaussian mean prior
v0 = 3e1; W0 = np.eye(D)/v0 # Wishart covariance prior

## Initialize parameters
m_k = []
W_k = []
beta_k  = np.repeat(beta0  + N/K, K)
alpha_k = np.repeat(alpha0 + N/K, K)
v_k     = np.repeat(v0 + N/K,     K)
for _ in range(K):
  # Let m_k be a random data point:
  m_k.append(data[np.random.choice(N)])
  # Let W_k be the mean of the Wishart prior:
  W_k.append(v0*W0)


## Loop until you're happy
max_iter = 100
ln_rho = np.zeros((N, K))
for iteration in range(max_iter):
  ## Variational E-step
  # XXX: FILL ME IN!
  
  ## Variational M-step
  # XXX: FILL ME IN!
  
## Plot data with distribution (we show expected distribution)
plt.figure(1)
plt.plot(data[:, 0], data[:, 1], '.')
for k in range(K):
  if Nk[k] > 0:
    plot_normal(m_k[k], np.linalg.pinv(v_k[k] * (W_k[k])))
plt.show()

## Now, animate the uncertainty by sampling
#num_samples = 100
#plt.figure(2)
#for s in range(num_samples):
#  plt.clf()
#  plt.plot(data[:, 0], data[:, 1], '.')
#  for k in range(K):
#    if Nk[k] > 0:
#      L = wishart.rvs(scale=W_k[k], df=v_k[k])
#      Sigma = np.linalg.pinv(L)
#      mu = multivariate_normal.rvs(mean=m_k[k], cov=Sigma/beta_k[k])
#      plot_normal(mu, Sigma)
#  plt.pause(0.01)

## Animate the entire mixture distribution
#fig = plt.figure(3)
#ax = fig.add_subplot(111, projection='3d')
#lim = np.linspace(-2, 2, 20)
#X, Y = np.meshgrid(lim, lim)
#XY = np.vstack((X.flatten(), Y.flatten())).T # 2500x2
#for s in range(num_samples):
#  pi_k = np.random.dirichlet(alpha_k)
#  Z = np.zeros(X.shape).flatten()
#  for k in range(K):
#    L = wishart.rvs(scale=W_k[k], df=v_k[k])
#    Sigma = np.linalg.pinv(L)
#    mu = multivariate_normal.rvs(mean=m_k[k], cov=Sigma/beta_k[k])
#    Z += pi_k[k] * multivariate_normal.pdf(XY, mean=mu, cov=Sigma)
#  ax.cla()
#  ax.plot_surface(X, Y, Z.reshape(X.shape))
#  ax.set_zlim(top=0.3)
#  plt.pause(0.01)

