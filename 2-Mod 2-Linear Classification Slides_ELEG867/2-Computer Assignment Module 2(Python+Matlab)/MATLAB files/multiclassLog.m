function [beta_matrix] = multiclassLog(train_feats, train_labels, num_labels)
% This function should implement the one-vs-all approach for multiclass
% classification using logistic regression

% Inputs: 
% train_feats: Training features
% train_labeles: Class labels of the training features
% num_labels: Number of class labels

% Outputs:
% beta_matrix: Matrix containing the beta coefficients for each classifier
% (should be arranged in a column-wise fashion) [beta_1 beta_2 .... beta_num_labels]

%=======Your code here======%
beta_matrix=zeros(num_labels,size(train_feats,2));

for idxLabel = 1:num_labels
% Here train your multilabel SVM
end

end
