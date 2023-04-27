function plotClass(beta, feats, labels)
% This function plots the data corresponding to 2 different classes and the
% decision boundary

% Inputs:
% Beta: Logistic regression coefficients
% fetas: Features
% labels: Class labels

figure(1); hold on;

% Find Indices of Positive and Negative Examples
pos = find(labels==1); neg = find(labels == 0);
% Plot Examples
plot(feats(pos, 2), feats(pos, 3), 'b+','LineWidth', 2, ...
'MarkerSize', 7);
plot(feats(neg, 2), feats(neg, 3), 'r+', 'MarkerFaceColor', 'r' ,'MarkerSize', 7);

hold on
% Only need 2 points to define a line, so choose two endpoints
x_lim = [min(feats(:,2))-1,  max(feats(:,2))+1];

% Calculate the decision boundary line
y_lim = (-1./beta(3)).*(beta(2).*x_lim + beta(1));

% Plot, and adjust axes for better viewing
plot(x_lim, y_lim,'k-')

% Legend, specific for the exercise
legend('Class 1', 'Class 0', 'Decision Boundary')
end
