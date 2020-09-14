import sys
import numpy as np
from codecs import encode
from training import train_mlp
from mlp import forward_propagation

def predicting(string_to_encode):
  alphabet = "abcdefghjiklmnopqrstuvxwyz"
  alphabet_characters = [char for char in alphabet]
  
  string_to_encode_characters = [char for char in string_to_encode]
  letter_positions = np.searchsorted(alphabet_characters, string_to_encode_characters)

  word = np.zeros((len(letter_positions), len(alphabet_characters)))
  for index, letter in enumerate(letter_positions):
    word[index][letter] = 1

  hidden_weights, hidden_bias, output_weights, output_bias = train_mlp()
  predicted, _ = forward_propagation(word, hidden_weights, hidden_bias, output_weights, output_bias)

  print("Original message:  %s" % string_to_encode)
  print("Predicted message: ", end='')
  for one_hot_encoded_letter in np.round(predicted):
    for encoded_letter, letter in zip(one_hot_encoded_letter, alphabet_characters):
      if encoded_letter and letter:
        print(letter, end='')

  print("\nRight message:\t   %s" % encode(string_to_encode, "rot_13"))

predicting(sys.argv[1])