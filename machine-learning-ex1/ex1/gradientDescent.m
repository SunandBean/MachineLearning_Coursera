function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    temp_sum_theta1 = 0;
    temp_sum_theta2 = 0;
    for idx_sum = 1:m
        temp_sum_theta1 = temp_sum_theta1 + (X(idx_sum,:) * theta - y(idx_sum)) * X(idx_sum , 1);
        temp_sum_theta2 = temp_sum_theta2 + (X(idx_sum,:) * theta - y(idx_sum)) * X(idx_sum , 2);
    end
    temp_theta_1 = theta(1) - alpha/m * temp_sum_theta1;
    temp_theta_2 = theta(2) - alpha/m * temp_sum_theta2;
    
    theta = [temp_theta_1;temp_theta_2];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
