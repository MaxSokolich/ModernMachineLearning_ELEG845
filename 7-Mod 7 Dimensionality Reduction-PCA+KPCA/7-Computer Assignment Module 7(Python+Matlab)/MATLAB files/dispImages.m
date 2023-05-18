function [image] = dispImages(train_feats)
% This functions displays example images
image_size=sqrt(size(train_feats(1,:),2));
image=[];
for i=1:8
    max_val=max(train_feats(i,:));
    image=[image reshape(train_feats(i,:)',image_size,image_size)/max_val];
end
imshow(image, [-1 1],'InitialMagnification','fit','Border','tight')
end

