%% Load Data

load('data_minst.mat')
X = train_feats(1:1000,:);
%% PCA
K = 100; % number of principal components

% Mean normalize features
[X_norm, mu] = normalize_features(X);

% PCA
[U, ~] = pca(X_norm);
%[U, S] = myeig(X_norm);

% Dimension reduction for face dataset
Z = project_data(X_norm, U, K);

% Recover data
X_rec  = recover_data(Z, U, K);

% Display normalized data
subplot(2, 1, 1);
im1=dispImages(X);
title('Original faces');

% Display reconstructed data from only k eigenfaces
subplot(2, 1, 2);
im2=dispImages(X_rec);
title('Recovered faces');