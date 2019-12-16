function J = symbolicJackie(intrinsicsMatrix, rotationVector, translationVector, points2d, points3d)
syms A R T
syms M m

jack = jacobian(A*R*M + A*T - m, [A, R, T]);

A = intrinsicsMatrix;
R = rotationVectorToMatrix(rotationVector);
T = translationVector';

M_all = points3d;
m_all = [points2d; ones(1, size(points2d,2))];

J = [];

for i = 1:size(M,2)
    M = M_all(:,i);
    m = m_all(:,i);
    J(i:i+2,:) = eval(subs(jack));
end

end