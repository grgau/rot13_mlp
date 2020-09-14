import math
import numpy as np
from mlp import mlp_start_weights, forward_propagation, backward_propagation, update_weights

def train_mlp():
  identity_n = 26
  encoder_input = np.identity(identity_n)
  encoder_output = np.roll(np.identity(identity_n), identity_n//2, axis=0)

  eta = 0.01
  stop_value = 0.0001
  max_iter = 500000
  mlp_input_hidden_output = (identity_n, int(math.log(identity_n,2)), identity_n)

  error_output_layer = math.inf
  input_size, hidden_size, output_size = mlp_input_hidden_output
  hidden_weights, hidden_bias, output_weights, output_bias = mlp_start_weights(input_size, hidden_size, output_size)

  for _ in range(max_iter):
    if error_output_layer < stop_value:
      break
    
    predicted, hidden_layer_output = forward_propagation(encoder_input, hidden_weights, hidden_bias, output_weights, output_bias)
    error = encoder_output - predicted
    derivative_predicted, derivative_hidden_layer = backward_propagation(error, predicted, hidden_layer_output, output_weights)
    output_weights, output_bias, hidden_weights, hidden_bias = update_weights(encoder_input, hidden_weights, hidden_bias, output_weights, output_bias, hidden_layer_output, derivative_predicted, derivative_hidden_layer, eta)
    error_output_layer = abs(error[-1][0])
  return hidden_weights, hidden_bias, output_weights, output_bias