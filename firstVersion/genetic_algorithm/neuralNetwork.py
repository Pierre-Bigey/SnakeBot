import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Randomly initialize weights
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))


    def forward(self, inputs):
        """Forward pass of the neural network."""
        # Compute hidden layer activations
        hidden = np.tanh(np.dot(inputs, self.weights_input_hidden))
        # Compute output layer activations
        output = np.tanh(np.dot(hidden, self.weights_hidden_output))
        return output
