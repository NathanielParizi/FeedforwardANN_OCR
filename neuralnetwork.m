%% Artificial Neural Network Image Classification 
% Nathaniel Parizi - MACHINE LEARNING  
%%
clc;
clear;
close all;

images = loadMNISTImages('training_set'); % initialize figure  
labels = loadMNISTLabels('training_label'); % initialize figure
labels = labels';   
labels(labels==0)=10;                               % dummyvar function represent single digit
labels=dummyvar(labels);                             

%%  Pattern Recognition Feedforward Neural Network   
net = patternnet(100, 'traingd');              % create Pattern Recognition Network
net.performFcn = 'mse';
t = labels';
x = images;
[net,tr] = train(net,x,t);                     % type of ANN, training data, labels

% define test, validation sets and performance metric.
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

%% Test for Accuracy
M = images(:,10000);
test_img = reshape(M, 28, 28);
figure
imshow(test_img);
predict = net(M) % output of class probabilities (Should show 7)
figure
bar(predict)