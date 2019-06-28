function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%[max_hidden_layer, index_hidden_layer] = max(sigmoid(X * Theta1'), [], 2);
input_layer_with_bias = [ones(m, 1) X];
hidden_layer = sigmoid(input_layer_with_bias * Theta1');
hidden_layer_column_length = size(hidden_layer, 1);
hidden_layer_with_bias = [ones(hidden_layer_column_length, 1) hidden_layer];
[max_output_layer, p] = max(sigmoid(hidden_layer_with_bias * Theta2'), [], 2);





% =========================================================================


end
