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

valueList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error = zeros(size(valueList,2),size(valueList,2));
count_c=1; count_sigma=1;

for c_aux = valueList,
  for sigma_aux = valueList,
    model = svmTrain(X, y, c_aux, @(x1, x2) gaussianKernel(x1, x2, sigma_aux));
    predictions = svmPredict(model, Xval);
    error(count_c, count_sigma) = mean(double(predictions ~= yval));
    count_sigma += 1 ;
  endfor
  count_c +=1;
  count_sigma = 1;
endfor

figure(2)
hold on;
for i=1:size(error,1),
  name = sprintf(";for C = %.2f;", valueList(i));
  plot(valueList,error(i,:), name);
  title("cross error vs sigma");
endfor

hold off;

minVal = min(min(error));
[C,sigma] = find(error == minVal);

%error

C = valueList(C);
sigma = valueList(sigma);

fprintf("C = %f and sigma = %f\n", C, sigma);
input("");

% =========================================================================

end
