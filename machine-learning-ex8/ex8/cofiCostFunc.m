function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



% R == 1 -> Calculate the cost
sum_J = 0;
for idx_movies = 1:num_movies
    for idx_users = 1:num_users
        if R(idx_movies,idx_users) == 1            
            sum_J = sum_J + (X(idx_movies,:)* Theta(idx_users,:)' - Y(idx_movies,idx_users))^2;
        end
        
    end
    
end

J = 1/2 * sum_J;

for idx_movies = 1:num_movies
    X_temp = X(idx_movies,:);
    sum_X_grad = 0;
    for idx_users = 1:num_users
        Theta_temp =Theta(idx_users,:);
        Y_temp = Y(idx_movies,idx_users);
        if R(idx_movies,idx_users) == 1 
            sum_X_grad = sum_X_grad + ((X_temp* Theta_temp')- Y_temp) * Theta_temp;
        end
    end
    X_grad(idx_movies,:) = sum_X_grad;
    X_grad(idx_movies,:) = X_grad(idx_movies,:) + lambda * X_temp;
end

for idx_users = 1:num_users
    Theta_temp = Theta(idx_users,:);
    
    sum_Theta_grad = 0;
    for idx_movies = 1:num_movies
        X_temp = X(idx_movies,:);
        Y_temp = Y(idx_movies,idx_users);
        if R(idx_movies,idx_users) == 1 
            sum_Theta_grad = sum_Theta_grad + ((X_temp* Theta_temp')- Y_temp) * X_temp;
        end
    end
    Theta_grad(idx_users,:) = sum_Theta_grad;
    Theta_grad(idx_users,:) = Theta_grad(idx_users,:) + lambda * Theta_temp;
end

theta2 = Theta.^2;
reg_J_theta = (lambda/2) * sum(theta2(:));

X2 = X.^2;
reg_J_X = (lambda/2) * sum(X2(:));

reg_J = reg_J_theta + reg_J_X;

J = J + reg_J;
Debug = 0;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
