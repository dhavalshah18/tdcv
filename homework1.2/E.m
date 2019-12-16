function [rho, W, error] = E(cameraParameters, rotationMatrix, translationVector, points3d, points2d)
% Points3d transpose?
est_points2d = project3d2image(points3d, cameraParameters, ...
    rotationMatrix, translationVector);

diff = (abs(points2d - est_points2d)).^2;
%error = diff(1:1:end);
error = [diff(1,:) diff(2,:)];

sigma = 1.48257968*mad(error,1);
error = error/sigma;

% Implement Tukey
c = 4.685;
in = abs(error)<c;
out = abs(error)>=c;

rho(in) = ((c^2)/6)*(1-(1-(error(in)/c).^2).^3);
rho(out) = (c^2)/6;

w(in) = (1-(error(in).^2/c^2)).^2;
w(out) = 0;
W = diag(w);
end