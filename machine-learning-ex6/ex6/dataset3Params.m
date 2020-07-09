function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Set_parameters = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

Set_PredictionError = zeros(64,1);

for idx_C = 1:8
    temp_C = Set_parameters(idx_C);
    for idx_sigma = 1:8
        temp_sigma = Set_parameters(idx_sigma);
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        predictions = svmPredict(model, Xval);
        Set_PredictionError(8*(idx_C-1)+idx_sigma,1) = mean(double(predictions ~= yval));
    end
    
end


[v, i] = min(Set_PredictionError)

idx_C_Fin = int8(i/8);
idx_sigma_Fin = i - 8*idx_C_Fin;

C = Set_parameters(idx_C_Fin);
sigma = Set_parameters(idx_sigma_Fin);

% =========================================================================

end
