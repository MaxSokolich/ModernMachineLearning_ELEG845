%% Generating random linear model
n=1000;
X = randn(n,1);
Beta_true = [2 3.5]; % only two nonzero coefficients

% Adding noise to the observations
Y = Beta_true(1) + X*Beta_true(2) + randn(n,1)*1.5; % small added noise

%% Using MATLAB
mdl = fitlm(X,Y,'linear');
Beta_ls(1)=table2array(mdl.Coefficients({'(Intercept)'},1));
Beta_ls(2)=table2array(mdl.Coefficients({'x1'},1));


%% Here you can create you own version of linear regression and compare the results
%Beta_myls=mylinreg(X,Y);
%% Plot
[Xsort,idx]=sort(X(:,1),'ascend');
Ysort=Y(idx);
scatter(Xsort,Ysort);
hold on
plot(Xsort,Beta_ls(1)+Beta_ls(2)*Xsort);
xlabel('x')
ylabel('y')