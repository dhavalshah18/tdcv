clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/tracking/valid/img';
% path to object ply file
object_path = '../data/teabox.ply';
% path to results folder
results_path = '../data/tracking/valid/results';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Create directory for results
if ~exist(results_path,'dir') 
    mkdir(results_path); 
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% TODO: setup camera parameters (camera_params) using cameraParameters()
fx = 2960.37845;
fy = fx;
cx = 1841.68855;
cy = 1235.23369;

intrinsicsMatrix = [fx 0 0;0 fy 0;cx cy 1];
camera_params = cameraParameters('IntrinsicMatrix',intrinsicsMatrix);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);

% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path
run('vlfeat/toolbox/vl_setup')

% Place SIFT keypoints and corresponding descriptors for all images here
keypoints = cell(num_files,1); 
descriptors = cell(num_files,1); 

% for i=1:length(Filenames)
%     fprintf('Calculating sift features for image: %d \n', i)
%     
% %    TODO: Prepare the image (img) for vl_sift() function
%     img = im2single(rgb2gray(imread(char(Filenames(i)))));
%     [keypoints{i}, descriptors{i}] = vl_sift(img) ;
% end

% Save sift features and descriptors and load them when you rerun the code to save time
% save('sift_descriptors.mat', 'descriptors')
% save('sift_keypoints.mat', 'keypoints')

load('sift_descriptors.mat');
load('sift_keypoints.mat');

%% Initialization: Compute camera pose for the first image

% As the initialization step for the tracking
% we need to compute the camera pose for the first image 
% The first image and it's camera pose will be our initial frame and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from the previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)

% Load the SIFT model from the previous task
load('sift_model.mat');

% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5 
% When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 2.0; 

% for i=1:num_files
%     fprintf('Calculating and matching sift features for image: %d \n', i)
%     
%     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch); 
% end


% % Save sift features, descriptors and matches and load them when you rerun the code to save time
% save('sift_matches.mat', 'sift_matches');

load('sift_matches.mat')

ransac_iterations = 2000; 
threshold_ransac = 750;
max_inliers = 4;
    
% Part i
 % Randomly select 4 correspondances from S
num_points = size(sift_matches{1}, 2);
sel = randperm(num_points, 4);
image_points = keypoints{1}(1:2, sift_matches{1}(1,sel))';
world_points = model.coord3d(sift_matches{1}(2,sel),:);

 % Estimate pose
[init_orientations,init_location,inlieridx,~] = ...
        estimateWorldCameraPose(image_points, world_points, ...
        camera_params, 'MaxReprojectionError', 10000, ...
        'MaxNumTrials', 10, 'Confidence', 0.0000001);


for r = 1:ransac_iterations
    fprintf('Running RANSAC iteration: %d \n', r)

    % Part ii
     % Reproject 3d points to 2d
    matches3d = model.coord3d(sift_matches{1}(2,:),:);
    points2d = keypoints{1}(1:2, sift_matches{1}(1,:));
    est_points2d = project3d2image(matches3d', camera_params, ...
        init_orientations, init_location);

     % Find Euclidean distance between points of image and reprojected
     % points
    diff = pdist2(points2d',est_points2d');
    diag_diff = diag(diff);
    inliers = (diag_diff <= threshold_ransac);
    inliers_ind = find(inliers)';

    % Part iii
    if (sum(inliers) > max_inliers)
        best_inliers = inliers_ind;
        max_inliers = sum(inliers);

    end

end

image_points = keypoints{1}(1:2, sift_matches{1}(1,best_inliers))';
world_points = model.coord3d(sift_matches{1}(2, best_inliers), :);

% TODO: Estimate camera position for the first image
[init_orientation, init_location,idx,status] = ...
    estimateWorldCameraPose(image_points, world_points, ...
    camera_params, 'MaxReprojectionError', 10);

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
%   Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;


% Method steps:
% 1) Back-project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1. 
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object
% 2) Find matches between descriptors of back-projected SIFT keypoints from the initial frame (image i) and the
% SIFT keypoints from the subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library
% 3) Project back-projected SIFT keypoints onto the subsequent frame (image i+1) using 3D coordinates from the
% step 1 and the initial camera pose 
% 4) Compute the reprojection error between 2D points of SIFT
% matches for the subsequent frame (image i+1) and 2D points of projected matches
% from step 3
% 5) Implement IRLS: for each IRLS iteration compute Jacobian of the reprojection error with respect to the pose
% parameters and update the camera pose for the subsequent frame (image i+1)
% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

% TODO: Implement IRLS method for the reprojection error optimisation
% You can start with these parameters to debug your solution 
% but you should also experiment with their different values
threshold_irls = 0.005; % update threshold for IRLS
N = 50; % number of iterations
threshold_ubcmatch = 1.5; % matching threshold for vl_ubcmatch()
coord = cell(num_files,1);

for i = 1:num_files-1
    fprintf('Running iteration: %d \n', i);
     % Step 1
    P = camera_params.IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) ...
        -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    
    %     Randomly select a number of SIFT keypoints
    perm = randperm(size(keypoints{i},2));
    sel = perm(1:30000);
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q;
    descriptors_new = [];
    
    for j = 1:30000
        m = [keypoints{i}(1:2,sel(j)); 1];
        lambda = norm(inv(Q)*m);
        r = orig + lambda*(inv(Q)*m);

        [~, t, u, v, coords] = TriangleRayIntersection(orig', (r-orig)', ...
            vertices(faces(:,1)+1,:), vertices(faces(:,2)+1,:), vertices(faces(:,3)+1,:));
        outliers = find(isnan(coords(:,1)));
        coords(outliers,:)=[];

        if ~isempty(coords)
            t(outliers,:)=[];
            [min_t, index_min] = min(t);
            coords = coords(index_min,:);
            coord{i} = [coord{i}; coords];
            descriptors_new = [descriptors_new, descriptors{i}(:,sel(j))];
        end
    end
     % Step 2
    sift_matches = vl_ubcmatch(descriptors_new, descriptors{i+1}, threshold_ubcmatch);
    
     % Step 3
    world_points = coord{i}(sift_matches(1,:),:);
    image_points = keypoints{i+1}(1:2, sift_matches(2,:));
    
    [init_orientations,init_locations,inlieridx,~] = ...
        estimateWorldCameraPose(image_points', world_points, ...
        camera_params, 'MaxReprojectionError', 20);

     % Step 4
     fprintf('EARL %d \n', i);
     [cam_in_world_orientations(:,:,i+1), cam_in_world_locations(:,:,i+1)] = ...
         earl(init_orientations, init_locations, camera_params, world_points, image_points, threshold_irls, N);
    
end



%% Plot camera trajectory in 3D world CS + cameras

load('good_rotations.mat')
load('good_translations.mat')
% load('good_rotations2.mat')
% load('good_translations2.mat')
figure()
% Predicted trajectory
visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, 'Color', 'b');
hold on;
% Ground Truth trajectory
visualise_trajectory(vertices, edges, gt_valid.orientations, gt_valid.locations, 'Color', 'g');
hold off;
title('\color{green}Ground Truth trajectory \color{blue}Predicted trajectory')

%% Visualize bounding boxes

figure()
for i=1:num_files
    
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    % Ground Truth Bounding Boxes
    points_gt = project3d2image(vertices',camera_params, gt_valid.orientations(:,:,i), gt_valid.locations(:, :, i));
    % Predicted Bounding Boxes
    points_pred = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points_gt(1, edges(:, j)), points_gt(2, edges(:,j)), 'color', 'g');
        plot(points_pred(1, edges(:, j)), points_pred(2, edges(:,j)), 'color', 'b');
    end
    hold off;
    
    filename = fullfile(results_path, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% Bonus part

% Save estimated camera poses for the validation sequence using Vision TUM trajectory file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
% In this task you should implement you own function to convert rotation matrix to quaternion

% Save estimated camera poses for the test sequence using Vision TUM 
% trajectory file format

% Attach the file with estimated camera poses for the test sequence to your code submission
% If your code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences
fileID = fopen('trajectory.txt', 'w');

for quack = 1:num_files
    quat = queeny(cam_in_world_orientations(:,:,quack));
    timestamp = 86400*(datenum(now) - datenum('01-Jan-1970 00:00:00') - 1/24);
    fprintf(fileID, '%f %f %f %f %f %f %f %f\r\n', timestamp, cam_in_world_locations(:,1,quack), cam_in_world_locations(:,2,quack), cam_in_world_locations(:,3,quack),...
        quat(1), quat(2), quat(3), quat(4));
    
end

fclose(fileID);