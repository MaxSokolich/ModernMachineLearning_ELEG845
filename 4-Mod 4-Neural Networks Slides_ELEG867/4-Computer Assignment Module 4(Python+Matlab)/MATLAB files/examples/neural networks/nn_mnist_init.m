function net = nn_mnist_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
% The first hidden layer with sigmoid activation function
% Since the input dimension is 28 * 28 and the first hidden layer has 32 neurons, the weights dimension is 28 * 28 * 1 * 32 
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(28,28,1,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sigmoid') ;

% The second hidden layer with sigmoid activation function
% Since the first hidden layer dimension is 32 * 1 and the second hidden layer has 16 neurons, thus the weights dimension is 1 * 1 * 32 * 16 
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,32,16, 'single'),zeros(1,16,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sigmoid') ; 

% The output layer
% Since the first hidden layer dimension is 16 * 1 and the output layer has 10 nodes, thus the weights dimension is 1 * 1 * 16 * 10 
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,16,10, 'single'),  zeros(1,10,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
% The cost/loss function                       
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
end

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = 0.005 ;
net.meta.trainOpts.numEpochs = 100 ;
net.meta.trainOpts.batchSize = 100 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
