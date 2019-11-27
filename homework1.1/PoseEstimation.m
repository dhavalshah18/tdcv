% @Department of Informatics , Technical University of Munich
% The code is second exercise of Keypoint based object detection, pose 
% estimation and refinement project in Tracking and Detection in Computer 
% Vision class, by Professor Ilic Slobodan.
% Students: Hossam Abdelhamid, Ferjad Naeem, and Borja Sánchez Clemente

clear;clc;
load('Matching.mat');
NumTestImages=size(matching,2); % Get test images numbers
prompt = 'What is threshold t ?: '; 
thres = input(prompt); % Threshold for Ransac
prompt = 'What is number of iterations N ?: ';
N = input(prompt); % Number of iterations
%% ================= Intrinsic Camera Parameters ======================
% Get test images directory
TestImageName= dir('data/images/detection/*.JPG'); 
% Read one image to obtain image size
ImName="data/images/detection/"+TestImageName(1).name;
TestImage=imread(convertStringsToChars(ImName));
ImSize=size(TestImage); % Get image size
f = [2960.37845 2960.37845]; % focal length
c = [1841.68855 1235.23369]; % Offset
K= [f(1) 0 c(1);0 f(2) c(2);0 0 1];
cameraParams=cameraIntrinsics(f,c,[ImSize(1) ImSize(2)]); % Intrinsic Param
%% ==================== Pose Estimation ==========================
Pose={};
for i=1:NumTestImages
  disp("Pose Estimation for Image: "+matching{i}.testname);
  num_points=size(matching{i}.Points2D,1); % Keypoints Number per image
  max=0; % maximum number of inliers per image
  inliers=[]; % inliers
  ind_inliers=[];  % inliers indices
  best_inliers=[]; % Best inliers indices
 for j=1:N
 % Randomly select a sample of 4 data points from S and estimate the pose using PnP
    selected = randsample(num_points,4); % Randomly select 4 points
    selected2D=matching{i}.Points2D(selected,:); % Randomly 2d points
    selected3D=matching{i}.Points3D(selected,:); % Randomly correspond 3d points
    [worldOrientation,worldLocation,inlierIdx, status] = estimateWorldCameraPose(...
    selected2D,selected3D,cameraParams, 'MaxNumTrials' ,10,'Confidence',...
    0.0000001,'MaxReprojectionError',1000);
    % Check error
    chk=sum(isnan(worldOrientation(:)))+sum(isnan(worldLocation));
    if(chk~=0)
       j=j-1;
       continue
    end
  % Determine the set of data points Si from all 2D-3D correspondences 
  % where reprojection error (Euclidean distance) is below the threshold t. 
  % The set Si is the consensus set of the sample and defines the inliers of S.  
    [R,t] = cameraPoseToExtrinsics(worldOrientation,worldLocation);
    P = K*[R'  t'];
    M=[matching{i}.Points3D ones(num_points,1)]' ;
    Est=P*M;
    Est_points2D=Est([1 2],:)./Est(3,:);
    diff=abs(matching{i}.Points2D'-Est_points2D);
    diff=diff.^2;
    sumd=sum(diff,1);
    sumd=sqrt(sumd);
    inliers=(sumd<=double(thres));
    ind_inliers=find(inliers);
    if(sum(inliers)>max)
    best_inliers=ind_inliers;
    max=sum(inliers);
    
    end
 end

 Pose{i}.TestName=matching{i}.testname; % Test image name
 % Inliers Keypoints in test image
 Pose{i}.TestInlierPoints2D=matching{i}.Points2D(best_inliers,:); 
 % correspoind 3D points in world coordinate for best model
 Pose{i}.InlierPoints3D=matching{i}.Points3D(best_inliers,:); 
  % Train image name
 Pose{i}.TrainName=matching{i}.trainname;
 % Inliers Keypoints in train image
 Pose{i}.TrainInlierPoints2D=matching{i}.trainkeypoints(best_inliers,:); 
 %Re-estimate the pose using Si and store it with the corresponding 
 %number of inliers.
 [worldOrientation,worldLocation,inlierIdx, status] = estimateWorldCameraPose(...
    Pose{i}.TestInlierPoints2D,Pose{i}.InlierPoints3D,cameraParams,...
    'MaxNumTrials' ,N,'Confidence',93,'MaxReprojectionError',thres);
    [R,t] = cameraPoseToExtrinsics(worldOrientation,worldLocation);
    P = K*[R'  t']; % Projection Matrix
    Pose{i}.ProjMatrix=P;
end
disp("done");
save('Pose.mat','Pose');