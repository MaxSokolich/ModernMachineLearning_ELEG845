function Z = project_data(X, U, K)
% project_data projects features X onto a K-dimensional space using the K
% eigenvectors associated with the K largest eigenvalues

% Input: 
% X: Features X
% U: Matrix of eigenvectors
% K: Number of principal components for the projection

% Outputs
% Z: Projected data

%% Please insert your code here
Z = zeros(size(X, 1), K);

end
