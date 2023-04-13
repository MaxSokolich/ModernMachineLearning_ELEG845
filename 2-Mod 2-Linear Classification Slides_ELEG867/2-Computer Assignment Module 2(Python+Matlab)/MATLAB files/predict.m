function p = predict(beta, feats)
%This function predicts whether the label is 0 or 1 using learned logistic 

prob = logistic(feats * beta);
prob(prob>=0.5)=1;
prob(prob<0.5)=0;
p=prob;

end
