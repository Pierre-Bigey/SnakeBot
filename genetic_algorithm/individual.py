import random

from SnakeBot.game import Game
from SnakeBot.utils.direction import Direction
from SnakeBot.genetic_algorithm.neuralNetwork import NeuralNetwork
import numpy as np


class Individual:
    def __init__(self, grid_size, hidden_size, game_iterations):
        """Initialize an individual whose genome represents the neural network weights."""
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.game_iterations = game_iterations

        input_size = grid_size * grid_size  # Flattened game state matrix
        output_size = 4  # Four possible directions: up, down, left, right

        self.neural_network = NeuralNetwork(input_size, hidden_size, output_size)

        # Flatten the network weights into a genome
        self.genome = np.concatenate([
            self.neural_network.weights_input_hidden.flatten(),
            self.neural_network.weights_hidden_output.flatten()
        ])
        self.fitness = 0

    def decode_genome(self):
        """Decode the genome back into the weights of the neural network."""
        input_hidden_size = self.neural_network.input_size * self.neural_network.hidden_size
        hidden_output_size = self.neural_network.hidden_size * self.neural_network.output_size

        # Ensure that the genome length matches the total size of both weight matrices
        assert len(self.genome) == input_hidden_size + hidden_output_size, "Genome size mismatch!"


        # Assign the genome back to the neural network weights
        self.neural_network.weights_input_hidden = np.reshape(self.genome[:input_hidden_size],
                                                              (self.neural_network.input_size,
                                                               self.neural_network.hidden_size))
        self.neural_network.weights_hidden_output = np.reshape(self.genome[input_hidden_size:],
                                                               (self.neural_network.hidden_size,
                                                                self.neural_network.output_size))

    def evaluate_fitness(self, grid_size, fitness_function):
        """Evaluate the fitness of this individual using the neural network to play the game."""
        self.decode_genome()

        fitness_sum = 0

        for _ in range(self.game_iterations):

            game = Game(grid_size)
            while not game.is_over:
                # Flatten the game state matrix and use it as input to the network
                game_state_matrix = game.get_state_matrix()
                flattened_state = np.array(game_state_matrix).flatten()

                # Use the neural network to decide the direction
                output = self.neural_network.forward(flattened_state)
                bot_direction = self.decode_output(output)
                game.update(bot_direction)

            # Compute the fitness
            fitness_sum += fitness_function(game)

        self.fitness = fitness_sum / self.game_iterations

    def decide_move(self, state):
        """Use the neural network to decide the next move."""
        flattened_state = np.array(state).flatten()
        output = self.neural_network.forward(flattened_state)
        return self.decode_output(output)

    def decode_output(self, output):
        """Convert the network's output into a movement direction."""
        direction_idx = np.argmax(output)  # Pick the direction with the highest value
        #Return the direction from Direction based on the index
        return Direction.from_index(direction_idx)

    def copy(self):
        """Create a copy of this individual."""
        new_individual = Individual(self.grid_size, self.hidden_size, game_iterations=self.game_iterations)
        new_individual.genome = self.genome.copy()
        new_individual.fitness = self.fitness
        return new_individual

    def mutate(self, mutation_rate = 0.01):
        """Mutate an individual's genome with a given probability."""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] += random.uniform(-0.1, 0.1)