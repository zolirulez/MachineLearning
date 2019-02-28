clear all
close all

%% Load data
%% We subsample the data, which gives us N pairs of (x, y)
load weather
x = (1:20:1000)';
y = TMPMAX(x);
N = numel(y);

%% Standardize data to have zero mean and unit variance
x = (x - mean(x)) ./ std(x); % Nx1
y = (y - mean(y)) ./ std(y); % Nx1

%% We want to predict values at x_* (denoted xs in the code)
M = 1000;
xs = linspace(min(x), max(x), M).'; % Mx1
%xs = linspace(-2, 2, M).'; % Mx1 --- try predicting over this interval instead

%% Initial kernel parameters -- once you have a GP regressor,
%% you should play with these parameters and see what happens
lambda0 = 100;
theta  = 2;

%% Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1).^2;

%% Compute covariance (aka "kernel") matrices
% XXX: FILL ME IN!
% K   = NaN(N, N); % NxN
% Ks  = NaN(N, M); % NxM
% Kss = NAN(M, M); % MxM
K = kernel(x,x,lambda0,theta) + sigma2*eye(N);
Ks = kernel(x,xs,lambda0,theta);
Kss = kernel(xs,xs,lambda0,theta);
warning('Compute covariance matrices')
 
%% Compute conditional mean p(y_* | x, y, x_*)
% XXX: FILL ME IN!
% mu = NaN(M, 1); %Mx1
% Sigma = NaN(M, M); % MxM
mu = Ks'/K*y;
Sigma = Kss-Ks'/K*Ks; 
warning('You should compute the conditional distribution!')

%% Plot the mean prediction
figure
plot(x, y, 'o-', 'markerfacecolor', 'k'); % raw data
hold all
plot(xs, mu); % mean prediction
hold off
title('Mean prediction');

%% Plot samples
figure
plot(x, y, 'ko', 'markerfacecolor', 'k'); % raw data
hold all
S = 500; % number of samples
samples = sample_gaussian(mu, Sigma, S); % SxM
for s = 1:S
  plot(xs, samples(s, :));
end % for
hold off
title('Samples');

%% Evaluate log-likelihood for a range of lambda's

Q = 100;
possible_lambdas = linspace(1, 300, Q);
loglikelihood = zeros(Q, 1);
for k = 1:Q
  lambda_k = possible_lambdas(k);
  warning('Compute log-likelihood of data for lambda_k');
  % XXX: FILL ME IN!
  K = kernel(x,x,possible_lambdas(k),theta) + sigma2*eye(N);
  loglikelihood(k) = -N/2*log(2*pi) - 1/2*logdet(K) - 1/2*y'/K*y;
end % for
[~, idx] = max(loglikelihood);
lambda_opt = possible_lambdas(idx);
figure
plot(possible_lambdas, loglikelihood)
hold on
plot(possible_lambdas(idx), loglikelihood(idx), '*')
hold off
title('Log-likelihood for \lambda');
xlabel('\lambda')

