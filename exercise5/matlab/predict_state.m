function new_state = predict_state(state, weights, delta_location, motor_noise)
num_particles = size(state, 1);

% new_state = NaN(num_particles, 2);

%% XXX: This function should draw samples from the predictive distribution
%%   p(state_t | state_{t-1}, delta_x, delta_y)
%% You need to implement this!
% STep 1
index = randsample(1:num_particles, 5000, true, weights);
% Step 2
new_state = state(index,:) + delta_location + motor_noise*randn(5000,2);

end % function
