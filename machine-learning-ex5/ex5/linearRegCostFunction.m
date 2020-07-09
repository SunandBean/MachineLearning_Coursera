function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


Temp_J = ((X * theta) - y).^2;


Temp_Theta = theta .* theta;
Temp_Theta(1) = 0;


J = (1/(2*m)) * sum(Temp_J) + (lambda/(2*m)) * sum(Temp_Theta);


Temp_Theta_grad = theta;
Temp_Theta_grad(1) = 0;

Temp_grad = (X * theta) - y;
Temp_grad = X' * Temp_grad;


grad = (1/m) * Temp_grad + (lambda/m) * Temp_Theta_grad;

% =========================================================================

grad = grad(:);

end
