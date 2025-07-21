
% Clear workspace and figures
clc;
clear;
close all;

%% Load Dataset
digitDatasetPath = fullfile('your_dataset_folder');  % <-- CHANGE THIS PATH
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Split Data
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% Data Augmentation
inputSize = [224 224 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXReflection', true);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

%% Load GoogleNet
net = googlenet;
lgraph = layerGraph(net);

%% Replace Final Layers
[learnableLayer, classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

%% Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train Network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%% Evaluate Model
[YPred, scores] = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

confMat = confusionmat(YValidation, YPred);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1 = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Average Precision: %.2f%%\n', mean(precision)*100);
fprintf('Average Recall: %.2f%%\n', mean(recall)*100);
fprintf('Average F1 Score: %.2f%%\n', mean(f1)*100);
