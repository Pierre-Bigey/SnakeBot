from view import WindowView
from game import Game
from utils.config import FPS
import time
from genetic_algorithm.ga_controller import GAController

def fitness_function(game):
    """Evaluate the fitness of a bot based on the game's score and survival time."""
    fitness = game.score   # Reward based on score (food eaten)
    return fitness

def main(render= False, grid_size= 20):

    grid_size = grid_size
    population_size = 20
    genome_length = 2
    generations = 100

    # Initialize the genetic algorithm controller
    ga_controller = GAController(
        population_size=population_size,
        genome_length=genome_length,
        fitness_function=fitness_function,
        grid_size=grid_size
    )

    # Run the genetic algorithm for a set number of generations
    best_individual = ga_controller.evolve(generations)

    # After evolving, we can display how the best individual performs
    print("Best Individual's Fitness:", best_individual.fitness)

    if not render:
        return best_individual

    # Optional: Watch the best individual play the game
    game = Game(grid_size)
    view = WindowView(game)
    while not game.is_over:
        # Let the best individual control the snake
        bot_direction = best_individual.decide_move(game.get_state())
        game.update(bot_direction)
        view.render()
        time.sleep(1 / FPS)


if __name__ == "__main__":
    main(render=True, grid_size=10)