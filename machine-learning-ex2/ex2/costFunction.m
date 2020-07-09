function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sum_j = 0;
for i = 1:m
    temp_Pos = (y(i)) * log(sigmoid(X(i,:)*theta));
    temp_Neg = (1 - y(i)) * log(1-sigmoid(X(i,:)*theta));
    sum_j = sum_j + (temp_Pos + temp_Neg);
end

J = -1/m * sum_j;

sum_dj = zeros(size(theta));

n = size(theta);
for i_n = 1:n
    for i_m = 1:m
%         sigmoid(X(i,:)) - y(i)
%         X(i,i_theta)
        sum_dj(i_n) = sum_dj(i_n) + (sigmoid(X(i_m,:)*theta) - y(i_m))*X(i_m,i_n);
    end
end
grad = 1/m * sum_dj;


% =============================================================

end
