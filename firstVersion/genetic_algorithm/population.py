from firstVersion.genetic_algorithm.individual import Individual


class Population:
    def __init__(self, size, grid_size, hidden_size, fitness_function, game_iterations):
        """Initialize the population with a list of individuals."""
        self.size = size
        self.fitness_function = fitness_function
        self.individuals = [Individual(grid_size, hidden_size, game_iterations) for _ in range(size)]
        self.grid_size = grid_size

    def evaluate_fitness(self):
        """Evaluate the fitness of each individual in the population."""
        for individual in self.individuals:
            # Make a copy of the game for each individual
            individual.evaluate_fitness(self.grid_size, self.fitness_function)

    def get_best_individual(self):
        """Return the individual with the highest fitness score."""
        return max(self.individuals, key=lambda ind: ind.fitness)
