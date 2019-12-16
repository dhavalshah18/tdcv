function [params] = f_lm(v,t,K,pts_2D,pts_3D,num_iter,thresh)
% Get rid of homogeneous coordinate
pts_2D = pts_2D(1:2,:);
pts_3D = pts_3D(1:3,:); 
params = [v t];
R = rotationVectorToMatrix(v)';
% estimated_2D = worldToImage(K,R,t,init_3D); % Reproject
% residual = init_2D - estimated_2D;
%% Jacobian
    function J = myjacobi(K,v,pts_2D,pts_3D)
        vx = skew_sym_mat(v);
        for i = 1:3
            e(1:3,i) = R(:,i); % ith base vector of R
            dRdv{i} = ( ( v(i)*vx + skew_sym_mat( cross(v,(eye(3)-R)*e(:,i) ) ) )/(norm(v)^2 ))*R;
        end
        dmtildedM = K;
        for M = 1: numel(pts_3D(1,:))
            dmdmtilde = [1 0 -pts_2D(1,M) ; 0 1 -pts_2D(2,M)];
            dMdp = [dRdv{1}*pts_3D(:,M) dRdv{2}*pts_3D(:,M) dRdv{3}*pts_3D(:,M) eye(3)];
            J(2*M-1:2*M,:) = dmdmtilde * dmtildedM * dMdp;
        end
    end


%% Implement Levenberg-Marquadt optimization algorithm

lambda = 0.001;
old_step = thresh + 1; % ensure step is initialized bigger than thresh
for i = 1:num_iter
    if old_step < thresh % If we move too slowly, we've converged already
        break
    end
    J = myjacobi(K,params(1:3),pts_2D,pts_3D);
    old_error = E(K,params,pts_3D,pts_2D);
    new_step = -inv(J'*J + lambda*eye(numel(J(1,:)))) * (J'*old_error);
    new_error = E(K,params + new_step,pts_3D,pts_2D);
    if new_error > old_error
        lambda = lambda * 10;
    else
        lambda = lambda / 10;
        params = params + new_step';
    end
    old_step = norm(new_step);
end
disp('Not converged');
end

function skewsym = skew_sym_mat(v)
skewsym = [0 -v(3) v(2) ; ...
    v(3) 0 -v(1) ; ...
    -v(2) v(1) 0];
end

function error = E(K,params,pts_3D,pts_2D)
R = rotationVectorToMatrix(params(1:3));
t = params(4:6);
% params = [v t]
P = K*[R'  t'];
M=[pts_3D ; ones(1,numel(pts_2D(1,:)))] ;
Est=P*M;
Est_points2D=Est([1 2],:)./Est(3,:); % divide by homogeneous coordinate
% sum of squared errors
dif=abs(pts_2D-Est_points2D);
% dif=dif.^2; not squared in Tukey's version
error=dif(1:1:end)'; % 1 row [x y x y x y...] to match J structure
%error=sum(dif,1); Do not sum x and y errors yet

% Implement Tukey
c = 4.685;
in = find(error<=c);
out = find(error>c);
rho(in) = c^2/6*(1-(1-(error(in)/c).^2).^3);
rho(out) = c^2/6;

%error = sum( abs( (K * [R' params(4:6)'] * pts_3D) - pts_2D ).^2 );
end