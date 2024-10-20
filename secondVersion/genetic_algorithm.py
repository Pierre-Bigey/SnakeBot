import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from copy import deepcopy
from neural_network import SnakeNet, choose_direction
from game import Game
from utils.config import *

matplotlib.use('TkAgg')

class GeneticAlgorithm:
    def __init__(self):
        self.population_size = POPUALTION_SIZE
        self.mutation_rate = MUTATION_RATE
        self.generations = GENERATIONS
        self.best_part = BEST_PART
        self.random_part = RANDOM_PART

        self.fitness_per_round = FITNESS_PER_ROUND
        self.fitness_per_food = FITNESS_PER_FOOD
        self.game_per_snake = GAME_PER_SNAKE

        # Initialize a population of SnakeNet models
        self.population = [SnakeNet() for _ in range(self.population_size)]

        #Plotting
        self.x = np.array([])
        self.y = np.array([])

        # enable interactive mode
        plt.ion()

        # creating subplot and figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line1, = self.ax.plot(self.x, self.y)

        # setting labels
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Updating plot...")


    def evaluate_fitness(self, model):
        """
        Evaluate the fitness of a given model by making it play several games and
        returning the average fitness score.
        """
        total_fitness = 0

        for _ in range(int(self.game_per_snake)):
            game = Game()

            while game.alive:
                # Choose direction based on the model
                direction = choose_direction(model, game)

                # Advance the game one step with the chosen direction
                alive, rounds, length = game.step(direction)

                if not alive:
                    break

            # Compute fitness based on the game outcome
            fitness = (length - 1) * self.fitness_per_food  # Each food gives a large bonus
            fitness += rounds * self.fitness_per_round  # Small bonus for each round survived
            total_fitness += fitness

        # Return the average fitness over multiple games
        return total_fitness / self.game_per_snake

    def mutate(self, model):
        """
        Apply random mutations to the weights of a model.
        """
        for param in model.parameters():
            if len(param.shape) > 1:  # Mutate weights, not biases
                mutation_tensor = torch.randn_like(param) * self.mutation_rate
                param.data += mutation_tensor

    def select_top_performers(self, population, fitnesses):
        """
        Select the top-performing models based on fitness scores.
        """
        top_count = int(self.population_size * self.best_part)
        sorted_indices = np.argsort(fitnesses)[::-1][:top_count]  # Get indices of top performers
        return [deepcopy(population[i]) for i in sorted_indices]

    def evolve_population(self):
        """
        Run the genetic algorithm for multiple generations.
        """
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")

            # Evaluate fitness for the current population
            fitnesses = [self.evaluate_fitness(model) for model in self.population]
            best_fitness = max(fitnesses)
            print(f"Best fitness: {best_fitness}\n")
            self.plot_new_point(best_fitness)

            # Select the top-performing individuals
            top_performers = self.select_top_performers(self.population, fitnesses)

            # Create the next generation
            next_population = []
            # First we keep the best performing models
            next_population.extend(top_performers)
            # Then we add some random models to maintain genetic diversity
            for _ in range(int(self.population_size * self.random_part)):
                model = SnakeNet()
                self.mutate(model)
                next_population.append(model)

            # Then we add mutated versions of the top performers
            while len(next_population) < self.population_size:
                parent = random.choice(top_performers)
                offspring = deepcopy(parent)
                self.mutate(offspring)
                next_population.append(offspring)

            # Replace the population with the new generation
            self.population = next_population

        # Return the best model from the final generation
        final_fitnesses = [self.evaluate_fitness(model) for model in self.population]
        best_index = np.argmax(final_fitnesses)
        return self.population[best_index]

    def plot_new_point(self, best_fitness):
        self.x = np.concatenate([self.x, [len(self.x)]])
        self.y = np.concatenate([self.y, [best_fitness]])

        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y)

        # update xlin and ylim
        self.ax.relim()
        self.ax.autoscale_view()

        # re-drawing the figure
        self.fig.canvas.draw()

        # to flush the GUI events
        self.fig.canvas.flush_events()