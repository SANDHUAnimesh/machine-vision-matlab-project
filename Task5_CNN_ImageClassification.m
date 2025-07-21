clc; clear; close all;

%% Load Digit Dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize images to 32x32 for LeNet
imds.ReadFcn = @(loc)imresize(imread(loc), [32 32]);

% Split into training (70%) and validation (30%)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Define LeNet-5 Architecture
layers = [
    imageInputLayer([32 32 1],'Name','input')

    convolution2dLayer(5,6,'Padding','same','Name','conv_1')
    averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1')

    convolution2dLayer(5,16,'Padding','same','Name','conv_2')
    averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')

    fullyConnectedLayer(120,'Name','fc_1')
    fullyConnectedLayer(84,'Name','fc_2')
    fullyConnectedLayer(10,'Name','fc_3')

    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

%% Define Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train Network
net = trainNetwork(imdsTrain, layers, options);

%% Classify and Evaluate Accuracy
YPred = classify(net, imdsValidation);
YVal = imdsValidation.Labels;

accuracy = sum(YPred == YVal)/numel(YVal);
fprintf("Accuracy: %.2f%%\n", accuracy * 100);
%% % Get predicted and true labels
YPred = classify(net, imdsValidation);
YTrue = imdsValidation.Labels;

% Confusion matrix
confMat = confusionmat(YTrue, YPred);

% Initialize metrics
numClasses = size(confMat,1);
precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1score = zeros(numClasses,1);

for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Average metrics
avg_precision = mean(precision);
avg_recall = mean(recall);
avg_f1 = mean(f1score);

% Display results
fprintf('\nAverage Precision: %.2f%%\n', avg_precision * 100);
fprintf('Average Recall: %.2f%%\n', avg_recall * 100);
fprintf('Average F1 Score: %.2f%%\n', avg_f1 * 100);

%% %% Load and Prepare Data
digitDatasetPath = fullfile('your_dataset_folder');  % Replace with your actual dataset path
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% Augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXReflection', true);

inputSize = [224 224 3];  % For GoogleNet
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

%% Load GoogleNet
net = googlenet;
lgraph = layerGraph(net);

% Replace final layers for transfer learning
[learnableLayer, classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

%% Training Options (20 Epochs)
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train Network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%% Evaluate Performance
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

