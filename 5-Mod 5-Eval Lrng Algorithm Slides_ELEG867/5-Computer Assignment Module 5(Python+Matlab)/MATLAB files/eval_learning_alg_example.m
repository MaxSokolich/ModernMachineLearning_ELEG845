%% Generate the data
n=5000; % Number of observations
rng(1)

X = 0.4*randn(n,1)+0.3;
Beta_true = [-2 1, 10 -5 -8 4]; % Define regression coefficients (5-order model)
% Adding noise to the observations
Y = Beta_true(1) + X(:,1)*Beta_true(2) + X(:,1).^2*Beta_true(3) + X(:,1).^3*Beta_true(4) + X(:,1).^4*Beta_true(5) + X(:,1).^5*Beta_true(6)+randn(n,1)*0.5; % small added noise

%% Divide number of observations into training testing and validation set
n_train=round(0.6*n);
n_val=round(0.1*n);
n_test=n-(n_train+n_val);

% X,Y Train
X_train=X(1:n_train);
Y_train=Y(1:n_train);
% X,Y Test
X_val=X(n_train+1:n_train+n_val);
Y_val=Y(n_train+1:n_train+n_val);
% X,Y Val
X_test=X(n_train+n_val+1:end);
Y_test=Y(n_train+n_val+1:end);
%% Using MATLAB
% polynomial fit
p=5; % Polynomial order for regression model
Beta_poly=polyfit(X_train,Y_train,p); % Find regression coefficient for a pth-order polynomial
Beta_poly=fliplr(Beta_poly); % Flip the order of the coefficients
%% Here you can create you own version of linear regression and compare the results
%[Beta_myls,J]=mypolreg(X,Y,p);
%% Plot regression results
[Xsort,idx]=sort(X_train(:,1),'ascend');
Ysort=Y_train(idx);
Yreg=0;
for i=1:p+1
    Yreg=Yreg+Beta_poly(i)*Xsort.^(i-1);
end
scatter(Xsort,Ysort);
hold on
plot(Xsort,Yreg);