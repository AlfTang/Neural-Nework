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

% Number of training examples
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

X = [ones(m,1) X]; % Add a column of 1's to the X matrix

% output from input layer
for i=1:m 
    a_hidden(:,i) = sigmoid(Theta1*X(i,:)'); 
end

a_hidden = [ones(m,1) a_hidden'];

% output from hidden layer
for j=1:m 
    h_theta(:,j) = sigmoid(Theta2*a_hidden(j,:)'); 
end

% Initialise the mapped y
yMapped = zeros(num_labels, m);

% Map the unrolled version of y to rolled one
for k = 1:m
    yMapped(y(k), k) = 1; 
end

%tic
%J = (-yMapped(:)' * log(h_theta(:)) - (1-yMapped(:))' * log(1-h_theta(:)))/m
%toc

Theta1NoBiasTerm = Theta1(:, 2:end);
Theta2NoBiasTerm = Theta2(:, 2:end);

%tic
J = sum(sum(-yMapped .* log(h_theta) - (1-yMapped) .* log(1-h_theta)))/m...
  + lambda/2/m*(Theta1NoBiasTerm(:)' * Theta1NoBiasTerm(:)... 
  + Theta2NoBiasTerm(:)' * Theta2NoBiasTerm(:));
%toc

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


% Compute gradient by backpropagation

% Step 2: Compute delta of output layer
delta_3 = h_theta - yMapped;

% Step 3: Compute delta of hidden layer
delta_2 = Theta2'*delta_3;
delta_2 = delta_2(2:end , :);
delta_2 = delta_2 .* sigmoidGradient(Theta1*X');

% Step 4: Compute Delta for every layer
Delta_1 = delta_2*X;
Delta_2 = delta_3*a_hidden;

% Step 5: Compute unregularized gradient by backpropagation
grad = [Delta_1(:); Delta_2(:)]/m;

% Step 6: Regularise gradient
Theta1(:,1) = 0;
Theta2(:,1) = 0;
grad = [Delta_1(:); Delta_2(:)]/m + lambda*[Theta1(:); Theta2(:)]/m;
end
