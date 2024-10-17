from genetic_algorithm.individual import Individual
from view import WindowView
from game import Game
from utils.config import FPS
import time
from genetic_algorithm.ga_controller import GAController

def fitness_function(game):
    """Evaluate the fitness of a bot based on the game's score and survival time."""
    fitness = game.score * 1000  # Reward based on score (food eaten)
    fitness += game.round  # Reward based on survival time
    return fitness

def main(render= False):

    grid_size = 5
    population_size = 60
    hidden_size = 16
    generations = 5000
    game_iterations = 15

    # Initialize the genetic algorithm controller
    ga_controller = GAController(
        population_size=population_size,
        grid_size=grid_size,
        hidden_size=hidden_size,
        fitness_function=fitness_function,
        game_iterations=game_iterations,
        generations=generations
    )

    # Run the genetic algorithm for a set number of generations
    best_individual = ga_controller.evolve()

    # After evolving, we can display how the best individual performs
    print("Best Individual's Fitness:", best_individual.fitness)

    #Save the best genome in a file named by date, hour, minute and fitness
    file = open("saves/best_genome_"+str(int(best_individual.fitness))+".txt", "w")
    file.write(str(best_individual.genome))

    if not render:
        return best_individual

    ready = input("Press Enter to watch the best individual play the game...")

    # Optional: Watch the best individual play the game
    game = Game(grid_size)
    view = WindowView(game)
    while not game.is_over:
        # Let the best individual control the snake
        state = game.get_state_matrix()
        bot_direction = best_individual.decide_move(state)
        game.update(bot_direction)
        view.render()
        time.sleep(1 / FPS)

def play_from_file(file_name):
    """Play a game using a genome saved in a file."""
    with open("saves\\"+file_name, 'r') as file:
        content = file.read()
    content = content.strip('[]')
    number_strings = content.split()
    genome = [float(num) for num in number_strings]

    grid_size = 5
    game = Game(grid_size)
    view = WindowView(game)
    individual = Individual(grid_size, 16, 20)
    individual.genome = genome
    individual.evaluate_fitness(grid_size, fitness_function)
    print("Loaded Individual's Fitness:", individual.fitness)
    while not game.is_over:
        state = game.get_state_matrix()
        bot_direction = individual.decide_move(state)
        game.update(bot_direction)
        view.render()
        time.sleep(1 / FPS)




if __name__ == "__main__":
    """If passed a file name as an argument, play a game using the genome saved in that file."""
    import sys
    if len(sys.argv) > 1:
        play_from_file(sys.argv[1])
    else:
        main(render=True)

