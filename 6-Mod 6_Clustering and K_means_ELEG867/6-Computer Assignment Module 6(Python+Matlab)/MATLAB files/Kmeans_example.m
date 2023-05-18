%% Generating data for K-means

% Generate three separate clusters
x1=1*randn(1000,2)+repmat([4,4],[1000,1]);
x2=1*randn(1000,2)+repmat([4,12],[1000,1]);
x3=1*randn(1000,2)+repmat([10,8],[1000,1]);

data=[x1;x2;x3];

%% Display the data
figure(1)
scatter(data(:,1),data(:,2))
title 'Data for clustering'

%% Perform K-means
[idx,C]= kmeans(data,3);

%% Uncomment this section to test your code
%C0=[4,1;4,7;12,7];
%[idx,C]= myKmeans(data,C0,3);
%% Plot results
figure(2);
plot(data(idx==1,1),data(idx==1,2),'r.','MarkerSize',12)
hold on
plot(data(idx==2,1),data(idx==2,2),'b.','MarkerSize',12)
hold on
plot(data(idx==3,1),data(idx==3,2),'g.','MarkerSize',12)
hold on
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off