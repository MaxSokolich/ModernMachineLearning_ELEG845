%% Generate 2 different classes using randomn data
n=2000;
train_size=1500;
rng('default')
rng(1) % seed
X1 = randn(n,1);
rng(2) % seed
X2= randn(n,1);

Beta_true1 = [2 3.5]; % only two nonzero coefficients
Beta_true2 = [2 3];

% Generate data for 2 classes using a linear model
rng(3) % seed
Y1 = Beta_true1(1) + X1*Beta_true1(2) + randn(n,1)*5+8; % small added noise
rng(4) % seed
Y2 = Beta_true2(1) + X2*Beta_true2(2) + randn(n,1)*5-8; % small added noise
%% Creating the dataset
% Training set and labels
train_feats=[X1(1:train_size) Y1(1:train_size);X2(1:train_size),Y2(1:train_size)];
train_labels=[ones(train_size,1);zeros(train_size,1)];
% Permuting elements in the training set
[idx_train_perm]=randperm(train_size*2);
train_feats(idx_train_perm,:)=train_feats;
train_labels(idx_train_perm)=train_labels;

% Testing set and labels
test_feats=[X1(train_size+1:end) Y1(train_size+1:end);X2(train_size+1:end),Y2(train_size+1:end)];
test_labels=[ones(n-train_size,1);zeros(n-train_size,1)];
%% Using MATLAB
train_feats=[ones(train_size*2,1) train_feats];
test_feats=[ones((n-train_size)*2,1) test_feats];
[beta,dev,stats] = glmfit(train_feats(:,2:3),train_labels,'binomial','logit');

%% Obtaining accuracy
p = predict(beta, test_feats);
fprintf('Test Accuracy: %f\n', mean(double(p == test_labels)) * 100);
%% Plot

plotClass(beta, test_feats, test_labels);
%% Here you can create you own version of logistic regression
%initial_beta = zeros(3, 1);
%[cost, grad] = costLogistic(initial_beta, train_feats, train_labels);
%options = optimset('GradObj', 'on', 'MaxIter', 400);
%[mybeta, cost] = fminunc(@(t)(costFunction(t, train_feats, train_labels)), initial_beta, options);