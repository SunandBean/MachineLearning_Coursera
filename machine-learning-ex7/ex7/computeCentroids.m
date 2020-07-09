function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

IdxSet_idx1 = find(idx == 1);
IdxSet_idx2 = find(idx == 2);
IdxSet_idx3 = find(idx == 3);

C_idx1 = length(IdxSet_idx1);
C_idx2 = length(IdxSet_idx2);
C_idx3 = length(IdxSet_idx3);

X_idx1 = X(IdxSet_idx1,:);
X_idx2 = X(IdxSet_idx2,:);
X_idx3 = X(IdxSet_idx3,:);

sum_X_idx1 = sum(X_idx1,1);
sum_X_idx2 = sum(X_idx2,1);
sum_X_idx3 = sum(X_idx3,1);

centroids(1,:) = 1/C_idx1 * sum_X_idx1;
centroids(2,:) = 1/C_idx2 * sum_X_idx2;
centroids(3,:) = 1/C_idx3 * sum_X_idx3;
% Debug = 0;





% =============================================================


end

