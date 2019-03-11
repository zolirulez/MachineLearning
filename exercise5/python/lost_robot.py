import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm

plt.figure()

def predict_state(state, weights, delta_location, motor_noise):
  num_particles = state.shape[0]
  new_state = np.zeros(state.shape)

  ## XXX: This function should return samples from the predictive distribution
  ##   p(state_t | state_{t-1}, delta_x, delta_y)
  ## You need to implement this!

def observe_landmark(true_location, landmarks, observation_noise):
  ## Determine nearest landmark (that's the one we observe)
  all_distances = np.sqrt(np.sum((landmarks -true_location)**2, axis=1))
  landmark_idx = np.argmin(all_distances)
  true_distance = all_distances[landmark_idx]
  
  ## Corrupt measurements by noise
  distance = np.abs(true_distance + observation_noise*np.random.randn())
  
  return (landmark_idx, distance)

## Define landmark positions (this is our map or the world)
landmarks = np.array([[-1, -1],
                      [-1,  1],
                      [ 1, -1],
                      [ 1,  1]]) # 4x2

## Noise characteristics of the system (try playing with these parameters)
motor_noise = 0.02 # std. dev. of the Gaussian noise on the robot motor
observation_noise = 0.2 # std. dev. of the Gaussian noise that corrupts the visual observation

## Define the motion the robot actually takes
## This is the location we want to estimate
## You don't need to understand this part of the code!
num_time_steps = 500
T = np.linspace(5*np.pi, np.pi, num_time_steps)
location = 2 * np.stack((T*np.cos(T), T*np.sin(T))).T/np.max(T) # (num_time_steps)x2

## Number of samples used in the particle filter
num_particles = 5000

## Initial state of the particle filter;
## state = (location), i.e. in R^2
state = np.zeros((num_particles, 2))
state[:, 0] = 6*np.random.rand(num_particles) - 3     # in [-3, 3]
state[:, 1] = 6*np.random.rand(num_particles) - 3     # in [-3, 3]
weights = np.ones(num_particles)/num_particles

## Iterate across time
estimated_location = np.zeros((num_time_steps, 2)) # (num_time_steps)x2
for t in range(num_time_steps):
  ## Extract the true position (we do not use this to determine the robot position)
  true_location = location[t]
  
  ## Noisy estimate of robot motion since last time step (in a real robot you would get this information from the motor control)
  if t == 0: # we treat first time step a bit differently
    delta_location = motor_noise*np.random.randn(2)
  else:
    delta_location = location[t] - location[t-1] + motor_noise*np.random.randn(2)
  
  ##### PREDICT
  ## The robot measures its relative motion since the last time step.
  ## This measurement is subject to noise.
  state = predict_state(state, weights, delta_location, motor_noise) # XXX: YOU NEED TO MODIFY THIS FUNCTION!
  
  ##### MEASURE
  (landmark_idx, distance) = observe_landmark(true_location, landmarks, observation_noise)
  lm = landmarks[landmark_idx]
  weights = np.zeros(num_particles) # XXX: YOU NEED TO DO THIS CORRECTLY!
  weights = weights / np.sum(weights)
  
  ## Compute current state mean
  mean_pos = weights.dot(state)
  estimated_location[t] = mean_pos
  
  ## Plot what's going on
  plt.clf()
  plt.plot(state[:, 0], state[:, 1], 'r.', markersize=10)
  plt.plot(true_location[0], true_location[1], 'g.', markersize=10)
  plt.plot(landmarks[:, 0], landmarks[:, 1], 'ko', markerfacecolor='k', markersize=10)
  plt.plot(mean_pos[0], mean_pos[1], 'b.', markersize=10)
  plt.plot(location[:, 0], location[:, 1], ':')
  plt.plot(estimated_location[:t, 0], estimated_location[:t, 1], '.-')
  plt.xlim(-2.2, 2.2)
  plt.ylim(-2.2, 2.2)
  plt.pause(0.05)

