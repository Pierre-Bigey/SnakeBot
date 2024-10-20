import sys

from genetic_algorithm import GeneticAlgorithm
from neural_network import *
from game import Game
from view import View

import time

def main():
    # Run the genetic algorithm to train a model
    ga = GeneticAlgorithm()
    ga.evolve_population()

def load(model_name):

    print(f"Loading the model from the file {model_name}")

    # Load the best model from file
    model = load_best_model(model_name)

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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        load("best_429.pth")
    else:
        main()
