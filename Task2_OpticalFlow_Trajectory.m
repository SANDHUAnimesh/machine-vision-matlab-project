%% %% Task 2 - Part 1: Corner Detection and Optical Flow
clear; clc; close all;

% Step 1: Load images and convert to grayscale
img1 = imread('GingerBreadMan_first.jpg');
img2 = imread('GingerBreadMan_second.jpg');
gray1 = rgb2gray(img1);
gray2 = rgb2gray(img2);

% Step 2: Detect corners in first image
corners = corner(gray1, 100);  % Max 100 corners
figure, imshow(img1); hold on;
plot(corners(:,1), corners(:,2), 'r*');
title('Corner Detection on GingerBreadMan\_first');

% Step 3: Estimate optical flow using opticalFlowLK
opticFlow = opticalFlowLK('NoiseThreshold',0.009);
flow1 = estimateFlow(opticFlow, gray1);
flow2 = estimateFlow(opticFlow, gray2);  % This gives us flow between img1 → img2

% Step 4: Visualise flow vectors on second image
figure;
imshow(img2); hold on;
plot(flow2, 'DecimationFactor', [5 5], 'ScaleFactor', 20);
title('Optical Flow from GingerBreadMan\_first to \_second');

%% Task 2 – Red Square Tracking
clear; clc; close all;

video = VideoReader('red_square_video.mp4');
frame = readFrame(video);
gray_prev = rgb2gray(frame);

%  Automatically detect best initial corner
corners = corner(gray_prev, 50);  % detect top 50 strongest corners
initial_point = corners(1, :);    % pick the strongest one (ranked)

track = initial_point;

opticFlow = opticalFlowLK('NoiseThreshold', 0.009);
estimateFlow(opticFlow, gray_prev);  % First frame init

while hasFrame(video)
    frame = readFrame(video);
    gray = rgb2gray(frame);

    % Find corners in this frame
    corners = corner(gray);
    last_point = track(end, :);

    % Find closest corner to last tracked position
    dists = vecnorm(corners - last_point, 2, 2);
    [~, idx] = min(dists);
    nearest_corner = corners(idx, :);

    % Estimate flow at that point
    flow = estimateFlow(opticFlow, gray);
    vx = flow.Vx(round(nearest_corner(2)), round(nearest_corner(1)));
    vy = flow.Vy(round(nearest_corner(2)), round(nearest_corner(1)));

    new_point = nearest_corner + [vx, vy];
    track = [track; new_point];
end

% Visualise on last frame
last_frame = frame;
imshow(last_frame); hold on;
plot(track(:,1), track(:,2), 'r-o', 'LineWidth', 2);
title('Improved Trajectory of Red Square');
%% %% Task 2 – Part III: RMSE and Comparison
load('red_square_gt.mat');  % loads groundtruth: gt_track_spatial

% Step 1: Align track lengths
estimated = track(2:end, :);                 % remove initial point
groundtruth = gt_track_spatial(1:end-1, :);  % adjust to match size

% Step 2: Compute RMSE
errors = estimated - groundtruth;
squared_error = sum(errors.^2, 2);
rmse = sqrt(mean(squared_error));
fprintf(' RMSE between estimated and ground truth trajectory: %.2f pixels\n', rmse);

% Step 3: Plot both trajectories
figure;
plot(groundtruth(:,1), groundtruth(:,2), 'g-o', 'LineWidth', 1.5); hold on;
plot(estimated(:,1), estimated(:,2), 'r--x', 'LineWidth', 1.5);
legend('Ground Truth', 'Estimated');
title('Trajectory Comparison: Ground Truth vs Estimated');
xlabel('X'); ylabel('Y'); grid on;
