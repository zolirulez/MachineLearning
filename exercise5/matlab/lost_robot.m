clear all
close all

figure
pause()

%% Define landmark positions (this is our map or the world)
landmarks = [-1, -1; ...
    -1,  1; ...
    1, -1; ...
    1,  1]; % 4x2

%% Noise characteristics of the system (try playing with these parameters)
motor_noise = 0.02; % std. dev. of the Gaussian noise on the robot motor
observation_noise = 0.2; % std. dev. of the Gaussian noise that corrupts the visual observation

%% Define the motion the robot actually takes
%% This is the location we want to estimate
%% You don't need to understand this part of the code!
num_time_steps = 500;
T = linspace(5*pi, pi, num_time_steps).';
location = 2.*[T.*cos(T), T.*sin(T)]./max(T); % (num_time_steps)x2

%% Number of samples used in the particle filter
num_particles = 5000;

%% Initial state of the particle filter;
%% state = (location), i.e. in R^2
state(:, 1) = 6*rand(num_particles, 1) - 3;     % in [-3, 3]
state(:, 2) = 6*rand(num_particles, 1) - 3;     % in [-3, 3]
weights = ones(1, num_particles)/num_particles; % (num_particles)x1

%% Iterate across time
estimated_location = NaN(num_time_steps, 2); % (num_time_steps)x2
for t = 1:num_time_steps
    %% Extract the true position (we do not use this to determine the robot position)
    true_location = location(t, :); % 1x2
    
    %% Noisy estimate of robot motion since last time step (in a real robot you would get this information from the motor control)
    if (t == 1)
        delta_location = motor_noise*randn(1, 2); % 1x2
    else
        delta_location = location(t, :) - location(t-1, :) + motor_noise*randn(1, 2); % 1x2
    end % if
    
    %%%%% PREDICT
    %% The robot measures its relative motion since the last time step.
    %% This measurement is subject to noise.
    state = predict_state(state, weights, delta_location, motor_noise); % XXX: YOU NEED TO MODIFY THIS FUNCTION!
    
    %%%%% MEASURE
    [landmark_idx, distance] = observe_landmark(true_location, landmarks, observation_noise);
    lm = landmarks(landmark_idx, :); % 1x2
    for j = 1:num_particles
        weights(j,:) = mvnpdf(norm(lm-state(j,:),2),distance,observation_noise); %NaN(num_particles, 1); % XXX: YOU NEED TO DO THIS CORRECTLY!
    end
    weights = weights / sum(weights); % (num_particles)x1
    
    %% Compute current state mean
    mean_pos = weights.' * state; % 1x2
    estimated_location(t, :) = mean_pos;
    
    %% Plot what's going on
    clf
    plot(state(:, 1), state(:, 2), 'r.', 'markersize', 10);
    hold on
    plot(true_location(1), true_location(2), 'g.', 'markersize', 10);
    plot(landmarks(:, 1), landmarks(:, 2), 'ko', 'markerfacecolor', 'k', 'markersize', 10);
    plot(mean_pos(1), mean_pos(2), 'b.', 'markersize', 10);
    plot(location(:, 1), location(:, 2), ':')
    plot(estimated_location(1:t, 1), estimated_location(1:t, 2), '.-')
    hold off
    axis([-2.2, 2.2, -2.2, 2.2]);
    pause(0.05)
end % for

