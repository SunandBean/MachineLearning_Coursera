function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sum_j = 0;
for i = 1:m
    temp_Pos = (y(i)) * log(sigmoid(X(i,:)*theta));
    temp_Neg = (1 - y(i)) * log(1-sigmoid(X(i,:)*theta));
    sum_j = sum_j + (temp_Pos + temp_Neg);
end

n = size(theta);

sum_thetasquare = 0;
for i_n = 2:n
    sum_thetasquare = sum_thetasquare + theta(i_n)*theta(i_n);
end

J = -1/m * sum_j + lambda * 1/(2*m) * sum_thetasquare;


sum_dj = zeros(size(theta));

for i_n = 1:n
    for i_m = 1:m
        sum_dj(i_n) = sum_dj(i_n) + (sigmoid(X(i_m,:)*theta) - y(i_m))*X(i_m,i_n);
    end
end

grad(1) = 1/m * sum_dj(1);
grad(2:end) = 1/m * sum_dj(2:end) + lambda * 1/m * theta(2:end);



% =============================================================

end
