import numpy as np

def forward_propagation(inputs, hidden_weights, hidden_bias, output_weights, output_bias):
  hidden_layer_activation = np.dot(inputs, hidden_weights) + hidden_bias
  hidden_layer_output = sigmoid_activation(hidden_layer_activation)
  output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
  predicted = sigmoid_activation(output_layer_activation)
  return predicted, hidden_layer_output

def backward_propagation(error, predicted, hidden_layer_output, output_weights):
  derivative_predicted = error * sigmoid_derivative(predicted)
  hidden_layer_error = derivative_predicted.dot(output_weights.T)
  derivative_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
  return derivative_predicted, derivative_hidden_layer

def update_weights(inputs, hidden_weights, hidden_bias, output_weights, output_bias, hidden_layer_output, derivative_predicted, derivative_hidden_layer, eta):
  output_weights += hidden_layer_output.T.dot(derivative_predicted) * eta
  output_bias += np.sum(derivative_predicted, axis=0, keepdims=True) * eta
  hidden_weights += inputs.T.dot(derivative_hidden_layer) * eta
  hidden_bias += np.sum(derivative_hidden_layer, axis=0, keepdims=True) * eta
  return output_weights, output_bias, hidden_weights, hidden_bias

def sigmoid_activation(x):
  return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def mlp_start_weights(mlp_input, mlp_hidden, mlp_output):
  hidden_weights = np.random.uniform(size=(mlp_input, mlp_hidden))
  hidden_bias = np.random.uniform(size=(1, mlp_hidden))
  output_weights = np.random.uniform(size=(mlp_hidden, mlp_output))
  output_bias = np.random.uniform(size=(1, mlp_output))
  return hidden_weights, hidden_bias, output_weights, output_bias