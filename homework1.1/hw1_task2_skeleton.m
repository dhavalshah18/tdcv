clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = './data/detection';
% path to object ply file
object_path = './data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Load the SIFT model from the previous task
load('sift_model.mat');


% TODO: setup camera intrinsic parameters using cameraParameters()
% Added these lines

focalLength = [2960.37845 2960.37845];
principalPoint = [1841.68855 1235.23369];
imageSize = [2456 3680];
camera_params = cameraIntrinsics(focalLength,principalPoint, imageSize);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Match SIFT features of new images to the SIFT model with features computed in the task 1
% You should use VLFeat function vl_ubcmatch()
run('vlfeat/toolbox/vl_setup')                                              % Added


% Place SIFT keypoints and descriptors of new images here
keypoints=cell(num_files,1);
descriptors=cell(num_files,1);
% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5 
% When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 2.0; 

% for i=1:num_files
%     fprintf('Calculating and matching sift features for image: %d \n', i)
%     
% %     TODO: Prepare the image (img) for vl_sift() function
%     img = im2single(rgb2gray(imread(char(Filenames(i)))));                  % Added
%     [keypoints{i}, descriptors{i}] = vl_sift(img);
% %     Match features between SIFT model and SIFT features from new image
%     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch); 
% end
% 
% 
% % Save sift features, descriptors and matches and load them when you rerun the code to save time
% save('sift_matches.mat', 'sift_matches');
% save('detection_keypoints.mat', 'keypoints')
% save('detection_descriptors.mat', 'descriptors')

load('sift_matches.mat')
load('detection_keypoints.mat')
load('detection_descriptors.mat')


%% PnP and RANSAC 
% Implement the RANSAC algorithm featuring also the following arguments:
% Reprojection error threshold for inlier selection - 'threshold_ransac'  
% Number of RANSAC iterations - 'ransac_iterations'

% Pseudocode
% i Randomly select a sample of 4 data points from S and estimate the pose using PnP.
% ii Determine the set of data points Si from all 2D-3D correspondences 
%   where the reprojection error (Euclidean distance) is below the threshold (threshold_ransac). 
%   The set Si is the consensus set of the sample and defines the inliers of S.
% iii If the number of inliers is greater than we have seen so far,
%   re-estimate the pose using Si and store it with the corresponding number of inliers.
% iv Repeat the above mentioned procedure for N iterations (ransac_iterations).

% For PnP you can use estimateWorldCameraPose() function
% but only use it with 4 points and set the 'MaxReprojectionError' to the
% value of 10000 so that all these 4 points are considered to be inliers

% Place camera orientations, locations and best inliers set for every image here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);
best_inliers_set = cell(num_files, 1);

ransac_iterations = 2000; 
%threshold_ransac = 700; % 700 with bad img 18, 20, 21, 22, saved as best_inliers_set700
threshold_ransac = 750; % bad 16, 18, 21, 22
% 
% for i = 1:num_files
%     fprintf('Running PnP+RANSAC for image: %d \n', i)
% %     TODO: Implement the RANSAC algorithm here
%     max_inliers = 4;
%     
%     % Part i
%      % Randomly select 4 correspondances from S
%     num_points = size(sift_matches{i}, 2);
%     sel = randperm(num_points, 4);
%     best_inliers = sel;
%     image_points = keypoints{i}(1:2, sift_matches{i}(1,sel))';
%     world_points = model.coord3d(sift_matches{i}(2,sel),:);
% 
%      % Estimate pose
%     [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i),inlieridx,status] = ...
%             estimateWorldCameraPose(image_points, world_points, ...
%             camera_params, 'MaxReprojectionError', 10000, 'MaxNumTrials', 10, 'Confidence', 0.0000001);
% 
% 
%     for r = 1:ransac_iterations
%         fprintf('Running RANSAC iteration: %d \n', r)
%         
%         % Part ii
%          % Reproject 3d points to 2d
%         matches3d = model.coord3d(sift_matches{i}(2,:),:);
%         points2d = keypoints{i}(1:2, sift_matches{i}(1,:));
%         est_points2d = project3d2image(matches3d', camera_params, ...
%             cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i));
%         
%          % Find Euclidean distance between points of image and reprojected
%          % points
%         diff = pdist2(points2d',est_points2d');
%         diag_diff = diag(diff);
%         inliers = (diag_diff <= threshold_ransac);
%         inliers_ind = find(inliers)';
%         
%         % Part iii
%         if (sum(inliers) > max_inliers)
%             best_inliers = inliers_ind;
%             max_inliers = sum(inliers);
% %             image_points = keypoints{i}(1:2, sift_matches{i}(1,best_inliers))';
% %             world_points = model.coord3d(sift_matches{i}(2, best_inliers), :);
% %             [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i),inlieridx,status] = ...
% %                 estimateWorldCameraPose(image_points, world_points, camera_params, ...
% %                 'MaxReprojectionError', 10);
% 
%         end
%         
%     end
%     
%     image_points = keypoints{i}(1:2, sift_matches{i}(1,best_inliers))';
%     world_points = model.coord3d(sift_matches{i}(2, best_inliers), :);
%     best_inliers_set{i} = best_inliers;
%     [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i),inlieridx,status] = ...
%         estimateWorldCameraPose(image_points, world_points, camera_params, ...
%         'MaxReprojectionError', 10, 'Confidence', 95);
%     
% end

% save('best_inliers_set750.mat', 'best_inliers_set');
% save('best_cam_locations750.mat', 'cam_in_world_locations')
% save('best_cam_orientations750.mat', 'cam_in_world_orientations')

% % Good detection with threshold 700
load('best_inliers_set700.mat')
load('best_cam_locations700.mat')
load('best_cam_orientations700.mat')

% % Good detection with threshold 750
% load('best_inliers_set750.mat')
% load('best_cam_locations750.mat')
% load('best_cam_orientations750.mat')

%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:num_files
    
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    
%   Plot inliers set
    %PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end