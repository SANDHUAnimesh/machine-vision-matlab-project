%% Task 1 - Part I: RGB Image Histogram
clear; clc; close all;

% Step 1: Read the image
img = imread('Dog.jpg');  % Replace with your image name
figure, imshow(img), title('Original RGB Image');

% Step 2: Separate RGB channels
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Step 3: Convert to double for plotting
R = double(R); G = double(G); B = double(B);

% Step 4: Plot Histograms for each channel
figure;
subplot(3,1,1);
histogram(R(:), 256, 'FaceColor', 'r'); title('Red Channel Histogram');

subplot(3,1,2);
histogram(G(:), 256, 'FaceColor', 'g'); title('Green Channel Histogram');

subplot(3,1,3);
histogram(B(:), 256, 'FaceColor', 'b'); title('Blue Channel Histogram');

%% Task 1 – Part II: Edge Detection
clear; clc; close all;

% Step 1: Read and convert to grayscale
img = imread('Dog.jpg');  % Replace with your image
gray_img = rgb2gray(img);
figure, imshow(gray_img), title('Original Grayscale Image');

% Step 2: Apply Sobel edge detection
sobel_edges = edge(gray_img, 'Sobel');
figure, imshow(sobel_edges), title('Sobel Edge Detection');

% Step 3: Apply Prewitt edge detection
prewitt_edges = edge(gray_img, 'Prewitt');
figure, imshow(prewitt_edges), title('Prewitt Edge Detection');

% Step 4: Apply Canny edge detection with default threshold
canny_edges = edge(gray_img, 'Canny');
figure, imshow(canny_edges), title('Canny Edge Detection (default)');

% Step 5: Try custom threshold for Canny
canny_edges_custom = edge(gray_img, 'Canny', [0.1 0.3]);  % Experiment here
figure, imshow(canny_edges_custom), title('Canny with Threshold [0.1 0.3]');

% Step 6: Add noise and re-test Sobel
noisy_img = imnoise(gray_img, 'salt & pepper', 0.02);
sobel_noisy = edge(noisy_img, 'Sobel');
figure, imshow(sobel_noisy), title('Sobel on Noisy Image');

% Step 7: Add Gaussian noise and test Prewitt
gaussian_noisy_img = imnoise(gray_img, 'gaussian', 0.01);
prewitt_noisy = edge(gaussian_noisy_img, 'Prewitt');
figure, imshow(prewitt_noisy), title('Prewitt on Gaussian Noisy Image');

