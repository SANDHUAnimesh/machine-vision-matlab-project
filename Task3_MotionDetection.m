%% Task 3: Moving Object Detection using GMM
clear; clc; close all;

% Step 1: Load video
video = VideoReader('car-tracking.mp4');

% Step 2: Create foreground detector using GMM
foregroundDetector = vision.ForegroundDetector( ...
    'NumGaussians', 3, ...
    'NumTrainingFrames', 50, ...
    'MinimumBackgroundRatio', 0.7);

% Step 3: Setup blob analysis to get bounding boxes
blobAnalysis = vision.BlobAnalysis( ...
    'BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, ...
    'CentroidOutputPort', false, ...
    'MinimumBlobArea', 200);

% Step 4: Loop through video
while hasFrame(video)
    frame = readFrame(video);
    fgMask = step(foregroundDetector, frame);

    % Step 5: Clean up using morphological operations
    fgMask = imopen(fgMask, strel('rectangle', [3,3])); % remove noise
    fgMask = imclose(fgMask, strel('rectangle', [15, 15])); % fill holes
    fgMask = imfill(fgMask, 'holes');

    % Step 6: Bounding boxes
bbox = step(blobAnalysis, fgMask);
result = insertShape(frame, 'Rectangle', bbox, 'Color', 'red');
result = insertText(result, [10, 10], ...
    sprintf('Cars Detected: %d (Config %d)', size(bbox, 1), i), ...
    'FontSize', 14, 'BoxColor', 'yellow');

%  Display both: mask + detection
subplot(1,2,1);
imshow(fgMask); title(sprintf('GMM Foreground Mask - Config %d', i));

subplot(1,2,2);
imshow(result); title(sprintf('Bounding Boxes - Config %d', i));

pause(0.01);

    % Step 7: Annotate video
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'red');
    result = insertText(result, [10, 10], ['Cars Detected: ', num2str(size(bbox, 1))], ...
        'FontSize', 14, 'BoxColor', 'yellow');
    

    imshow(result); title('Detected Moving Objects');
    pause(0.01);
end