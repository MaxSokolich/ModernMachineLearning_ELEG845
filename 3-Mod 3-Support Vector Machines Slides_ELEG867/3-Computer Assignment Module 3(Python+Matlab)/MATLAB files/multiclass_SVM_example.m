
% Load Training Data
load('data_minst');
train_num=4500;
m = size(train_feats, 1);
input_layer_size  = size(train_feats, 2); % Should be 784 for 28 x 28 images
num_labels = max(train_labels); % Should be 10, the number of digits     

% Display example images
dispImages(train_feats)

% Adding the bias term
test_feats = [ones(m-train_num, 1) train_feats(train_num+1:end,:)];
test_labels=train_labels(4500+1:end);
train_feats = [ones(train_num, 1) train_feats(1:train_num,:)];
train_labels= train_labels(1:train_num);

%% Multiclass classification (your code)

% Create your own function multiclassSVM() below to perform one vs all 
% classification for the Extended YaleB face database
v = multiclassSVM(train_feats, train_labels, test_feats, num_labels);
%% Calculating accuracy

fprintf('\n Accuracy: %f\n', mean(double(v == test_labels)) * 100);