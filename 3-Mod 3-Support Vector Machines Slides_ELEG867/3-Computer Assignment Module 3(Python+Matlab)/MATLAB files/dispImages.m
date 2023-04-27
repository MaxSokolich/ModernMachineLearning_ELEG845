function [] = dispImages(train_feats)
% This functions displays example images
image_size=sqrt(size(train_feats(1,:),2));
image=[];
for i=1:5
    max_val=max(train_feats(i,:));
    image=[image reshape(train_feats(i,:)',image_size,image_size)/max_val];
end
imshow(image, [-1 1],'InitialMagnification','fit','Border','tight')
set(gca,'units','pixels') % set the axes units to pixels
x = get(gca,'position'); % get the position of the axes
set(gcf,'units','pixels'); % set the figure units to pixels
y = get(gcf,'position'); % get the figure position
set(gcf,'position',[y(1) y(2) x(3) x(4)])% set the position of the figure to the length and width of the axes
set(gca,'units','normalized','position',[0 0 1 1]) % set the axes units to pixels
end

