function [R, t] = earl(rotationMatrix, translationVector, cameraParameters, points3d, points2d, threshIRLS, numiter)
%ENERGY_FUNCTION Summary of this function goes here
%   rotationParameters: rotation matrix
%   translationParameters: translation matrix
%   cameraParameters: cameraParameters type input argument describing
%       intrinsics matrix
%   points3d: Mi, 3D points corresponding to 2d points
%   points2d: mi, 2D points corresponding to 3d points

% Something about rodrigues formula
%rotationParameters = rotationMatrixToVector(rotationMatrix);+
% Don't know if we need transpose or not?
%cameraParameters.IntrinsicMatrix = cameraParameters.IntrinsicMatrix';

rotationVector = rotationMatrixToVector(rotationMatrix);

points3d = points3d';

lambda = 0.00001;
old_delta = threshIRLS + 1; % ensure step is initialized bigger than thresh

for i = 1:numiter
    fprintf('IRLS Iteration %d\n', i);
    if old_delta < threshIRLS % If we move too slowly, we've converged already
        break
    end
    
    J = jack(cameraParameters.IntrinsicMatrix', rotationVector, points2d, points3d);
    
    [RHO, W, error] = E(cameraParameters, rotationMatrix, translationVector, points3d, points2d);
    sum_RHO = sum(RHO);
    
    delta = -inv(J'*W*J + lambda*eye(size(J,2))) * (J'*W*error');
    
    [new_error, ~] = E(cameraParameters, rotationVectorToMatrix(rotationVector + delta(1:3)'),...
        translationVector + delta(4:6)', points3d, points2d);
    sum_new_error = sum(new_error);
    
    if sum_new_error > sum_RHO
        lambda = lambda * 10;
    else
        fprintf("ELSE\n");
        lambda = lambda / 10;
        rotationMatrix = rotationVectorToMatrix(rotationVector + delta(1:3)');
        translationVector = translationVector + delta(4:6)';
    end
    old_delta = norm(delta);
end

R = rotationMatrix;
t = translationVector;

end




