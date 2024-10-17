from SnakeBot.game import Game
from SnakeBot.genetic_algorithm.individual import Individual


class Population:
    def __init__(self, population_size, genome_length, fitness_function, grid_size):
        """Initialize the population with a list of individuals."""
        self.population_size = population_size
        self.genome_length = genome_length
        self.fitness_function = fitness_function
        self.individuals = [Individual(genome_length) for _ in range(population_size)]
        self.grid_size = grid_size

    def evaluate_fitness(self):
        """Evaluate the fitness of each individual in the population."""
        for individual in self.individuals:
            # Make a copy of the game for each individual
            game_copy = Game(self.grid_size)  # Reset game for each bot
            individual.evaluate_fitness(game_copy, self.fitness_function)

    def get_best_individual(self):
        """Return the individual with the highest fitness score."""
        return max(self.individuals, key=lambda ind: ind.fitness)
