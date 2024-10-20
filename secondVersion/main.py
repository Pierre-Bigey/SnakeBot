import sys

from genetic_algorithm import GeneticAlgorithm
from neural_network import *
from game import Game
from view import View

import time

import cProfile

def main():
    # Run the genetic algorithm to train a model
    with cProfile.Profile() as pr:
        ga = GeneticAlgorithm()
        ga.evolve_population()

def load(model_name):

    print(f"Loading the model from the file {model_name}")

    # Load the best model from file
    model = load_best_model(model_name)

    ga = GeneticAlgorithm(False)
    ga.game_per_snake = GRID_SIZE * GRID_SIZE
    fitness = ga.evaluate_fitness(model)

    print(f"Fitness of the model: {fitness}")

    # Create a new game and view
    game = Game()
    view = View()

    view.render(game)

    # Play the game using the best model
    while game.alive:
        direction = choose_direction(model, game)
        game.step(direction)
        view.render(game)
        time.sleep(1/FPS)

    print(f"Game over! Score: {len(game.snake) - 1}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        load("best_11870.pth")
    else:
        main()
