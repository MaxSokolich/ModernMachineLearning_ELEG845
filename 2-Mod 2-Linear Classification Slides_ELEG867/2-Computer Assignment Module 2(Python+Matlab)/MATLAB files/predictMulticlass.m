function pred = predictMulticlass(beta_matrix, test_feats)
%Predict the label for a multilabel classifier 

% Inputs:
% beta_matrix: Matrix containing the different classifiers beta_j (column-wise)
% teast_feats: Testing examples

% Outputs: 
% pred: Predicted labels

%% 
m = size(test_feats, 1);

% Vector containing predictions 
pred = zeros(size(test_feats, 1), 1);

for idxSample=1:m
    % Estimates a vector which ten probabilities corresponding to 10 classes using sigmoid(probability function)
    testSig=logistic(beta_matrix*test_feats(idxSample,:)'); 
    pred(idxSample)=find(testSig==max(testSig));
end
end
