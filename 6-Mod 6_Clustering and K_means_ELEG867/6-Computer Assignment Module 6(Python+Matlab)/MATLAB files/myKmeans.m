function [idx,C] = myKmeans(X,C0,K)
% This functions performs K-means. You should write the functions
% cluster_assignment, and cluster update. You can also modify this script
% to do any of the tasks given in Computer Assignment #6

%Inputs
% X: Data
% C0: Initial position of the centrois, given row-wise
% K: Number of centrois

% Outputs%
% idx: Cluster assigned to every point in X
% C: Position of the centroids

%% Your code starts here
C=C0;
max_iters=30;

for i=1:max_iters
    % Assign clusters
    idx = cluster_assignment(X,C,K);    
    % Update the positions of the clusters
    C = cluster_update(X, idx, K);
end

end

