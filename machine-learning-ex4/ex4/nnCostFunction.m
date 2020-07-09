function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1
A1 = [ones(m, 1) X];

A2 = sigmoid(A1*Theta1');

A2 = [ones(size(A2,1), 1) A2];
A3 = sigmoid(A2*Theta2');

J_temp = 0;

for idx_m = 1:m
    X_temp = A3(idx_m,:)';
    y_temp = zeros(num_labels,1);
    y_temp(y(idx_m)) = 1;
    
    J_temp = J_temp + sum(y_temp.*log(X_temp) + (1-y_temp).*(log(1-X_temp)));
end


% wo Reg term
J =  (-1/m) * J_temp;

% w Reg term
temp_Theta1 = Theta1(1:hidden_layer_size, 1+1:input_layer_size+1);
temp_Theta2 = Theta2(1:num_labels, 1+1:hidden_layer_size+1);
Reg_theta1 = sum(sum(temp_Theta1.*temp_Theta1));
Reg_theta2 = sum(sum(temp_Theta2.*temp_Theta2));
Reg = Reg_theta1 + Reg_theta2;

J = J + (lambda/(2*m)) * Reg;

% Part 2

Z2 = A1*Theta1';
Z2 = [ones(size(Z2,1), 1) Z2];
Z3 = A2*Theta2';

delta_3 = zeros(num_labels,1);
for idx_m = 1:m
    A3_temp = A3(idx_m,:)';
    y_temp = zeros(num_labels,1);
    y_temp(y(idx_m)) = 1;
    
    for idx_k = 1:num_labels
        delta_3(idx_k) = (A3_temp(idx_k) - y_temp(idx_k));
    end
    
    delta_2 = Theta2'* delta_3.*sigmoidGradient(Z2(idx_m,:)');
    delta_2 = delta_2(2:end);
    
    Theta2_grad = Theta2_grad + delta_3*A2(idx_m,:);
    Theta1_grad = Theta1_grad + delta_2*A1(idx_m,:);
    Debug=0;
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;

Theta1_grad = Theta1_grad + (lambda/m) * Theta1_reg;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2_reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
