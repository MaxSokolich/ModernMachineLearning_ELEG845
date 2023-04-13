% Generating observation
X = randn(100,8);

% True values of Beta that will be later estimated using lasso
Beta_true = [0;2;0;-3;0;5;0;10]; % only two nonzero coefficients

% Adding noise to the observations
Y = X*Beta_true + randn(100,1)*.1; % small added noise

% Using MATLAB
Beta_lasso = lasso(X,Y,'lambda',1e-3); %vary lambda from 1e-3 to 10
%% Here you can create you own version of lasso and compare the results
%lambda=1e-3;
%Beta_mylasso=mylasso(X,Y,lambda);
%%