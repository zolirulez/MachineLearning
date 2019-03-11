function [landmark_idx, distance] = observe_landmark(true_location, landmarks, observation_noise)
%% Determine nearest landmark (that's the one we observe)
all_distances = sqrt(sum(bsxfun(@minus, landmarks, true_location).^2, 2)); % 4x1
[true_distance, landmark_idx] = min(all_distances);
lm = landmarks(landmark_idx, :); % 1x2

%% Corrupt measurements by noise
distance = abs(true_distance + observation_noise*randn());
end % function
