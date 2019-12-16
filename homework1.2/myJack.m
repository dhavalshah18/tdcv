function [J] = myJack(intrinsicsMatrix, rotationVector, translationVector, real_points2d, real_points3d)
syms u0 v0 f;
syms wx wy wz tx ty tz;
syms X Y Z;
syms x y;

A = [f 0 u0; 0 f v0; 0 0 1]; % pixel size?

w = [wx wy wz];
theta = norm(w);
omega = [0 -wz wy; wz 0 -wx; -wy wx 0]; %use our function skew_sym_matrix here
R = eye(3)+(sin(theta)/theta)*omega+((1-cos(theta))/theta^2)*omega^2;

T = [tx ty tz]';
points3d = [X Y Z]';
points2d = [x y];

est_points2d_h = A*(R*points3d + T);
est_points2d = [est_points2d_h(1)/est_points2d_h(3) est_points2d_h(2)/est_points2d_h(3)];
% use the function here

du = abs(points2d(1)-est_points2d(1));
dv = abs(points2d(2)-est_points2d(2));

J1 = [diff(du,wx) diff(du,wy) diff(du,wz)...
    diff(du,tx) diff(du,ty) diff(du,tz);...
    diff(dv,wx) diff(dv,wy) diff(dv,wz)...
    diff(dv,tx) diff(dv,ty) diff(dv,tz)];

points2d = real_points2d;
points3d = real_points3d;
X = points3d(1,1); Y = points3d(1,2); Z = points3d(1,3);
x = points2d(1,1); y = points2d(1,2);

wx = rotationVector(1); wy = rotationVector(2); wz = rotationVector(3);
tx = translationVector(1); ty = translationVector(2); tz = translationVector(3);

A = intrinsicsMatrix;
f = A(1,1); u0 = A(1,3); v0 = A(2,3);

J = eval(subs(J1));

end

