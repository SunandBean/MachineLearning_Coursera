function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for idx_X = 1:size(X,1)
    X_Cur = X(idx_X,:);
    c_temp = zeros(K,1);
    for idx_C = 1:K
        C_Cur = centroids(idx_C,:);
        c_temp(idx_C,1) = sqrt(sum((X_Cur - C_Cur).^2));
    end
%     Debug = 0;
    [~, idx(idx_X,1)] = min(c_temp);
%     Debug = 0;
end

% Debug = 0;



% =============================================================

end

