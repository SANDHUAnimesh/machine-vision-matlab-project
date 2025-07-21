% Robot Treasure Hunt - Final Polished Version for Submission
clc;
clear;
close all;

% Load Image (Easy Level)
im = imread('Treasure_easy.jpg');
figure('Name', 'Original Image'); imshow(im); title('Original RGB Image');

% Binarisation
bin_threshold = 0.1;
bin_im = im2bw(im, bin_threshold);
figure('Name', 'Binarized Image'); imshow(bin_im); title('Binarized Image');

% Connected Components
con_com = bwlabel(bin_im);
figure('Name', 'Connected Components'); imshow(label2rgb(con_com)); title('Connected Components');

% Object Properties
props = regionprops(con_com);

% Draw Bounding Boxes
figure('Name', 'Objects with Bounding Boxes'); imshow(im); title('Bounding Boxes'); hold on;
for i = 1:numel(props)
    rectangle('Position', props(i).BoundingBox, 'EdgeColor', 'b');
end
hold off;

% Find yellow dots
yellow_props = find_yellow_dots(im);

% Detect arrows from yellow dots
arrow_indices = find_arrows(props, yellow_props);

% Detect red arrow (start)
red_arrow_id = find_red_arrow(arrow_indices, props, im);

% Run treasure hunt logic
[path, treasure_id] = hunt_treasure(red_arrow_id, props, yellow_props, im, arrow_indices, con_com);

% Visualise path and final treasure
visualize_path_and_treasure(im, props, path, treasure_id);

% ---- Subfunctions ----

function yellow_props = find_yellow_dots(im)
    yellow_mask = im(:,:,1) > 150 & im(:,:,2) > 200 & im(:,:,3) < 100;
    yellow_labels = bwlabel(yellow_mask);
    yellow_props = regionprops(yellow_labels);
end

function arrow_indices = find_arrows(props, yellow_props)
    arrow_indices = [];
    for r = 1:numel(props)
        for s = 1:numel(yellow_props)
            if check_object_in_bbox(yellow_props(s).Centroid, props(r).BoundingBox)
                arrow_indices(end+1,:) = [r, s];
                break;
            end
        end
    end
end

function red_arrow_id = find_red_arrow(arrow_indices, props, im)
    red_arrow_id = [];
    for arrow_info = arrow_indices'
        arrow_id = arrow_info(1);
        color = squeeze(im(round(props(arrow_id).Centroid(2)), round(props(arrow_id).Centroid(1)), :));
        if color(1) > 240 && color(2) < 10 && color(3) < 10
            red_arrow_id = arrow_id;
            return;
        end
    end
    error('Red arrow not found');
end

function inside = check_object_in_bbox(centroid, bbox)
    inside = centroid(1) > bbox(1) && centroid(1) < (bbox(1)+bbox(3)) && ...
             centroid(2) > bbox(2) && centroid(2) < (bbox(2)+bbox(4));
end

function [path, treasure_id] = hunt_treasure(start_arrow_id, props, yellow_props, im, arrow_indices, con_com)
    path = [];
    treasure_id = [];
    cur = start_arrow_id;
    done = false;
    while ~done
        while ismember(cur, arrow_indices(:,1))
            path(end+1) = cur;
            cur = find_next_object(cur, path, props, yellow_props, im, arrow_indices, treasure_id, con_com);
        end
        treasure_id(end+1) = cur;
        [cur, done] = find_next_object(cur, path, props, yellow_props, im, arrow_indices, treasure_id, con_com);
    end
end

function [next_object, done] = find_next_object(cur_object, path, props, yellow_props, im, arrow_indices, treasure_id, con_com)
    next_object = [];
    done = false;

    arrow_info = arrow_indices(arrow_indices(:,1) == cur_object, :);
    if isempty(arrow_info) || arrow_info(1,2) > numel(yellow_props)
        done = true; return;
    end

    yellow_centroid = yellow_props(arrow_info(1,2)).Centroid;
    arrow_centroid = props(cur_object).Centroid;
    dir = yellow_centroid - arrow_centroid;
    dir = dir / norm(dir);

    for step = 1:200
        probe = arrow_centroid + dir * step * 2;
        x = round(probe(1)); y = round(probe(2));
        if x < 1 || y < 1 || x > size(im,2) || y > size(im,1)
            break;
        end
        label = con_com(y, x);
        if label ~= 0 && label ~= cur_object && ~ismember(label, [path, treasure_id])
            next_object = label;
            return;
        end
    end
    done = true;
end

function visualize_path_and_treasure(im, props, path, treasure_id)
    figure('Name', 'Treasure Path');
    imshow(im); title('Robot Path to Treasure'); hold on;
    for i = 1:numel(path)
        bb = props(path(i)).BoundingBox;
        rectangle('Position', bb, 'EdgeColor', 'yellow', 'LineWidth', 2);
        text(bb(1)+bb(3)/2, bb(2)-5, num2str(i), 'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold');
    end
    for i = 1:numel(treasure_id)
        rectangle('Position', props(treasure_id(i)).BoundingBox, 'EdgeColor', 'green', 'LineWidth', 3);
    end
    hold off;
end
