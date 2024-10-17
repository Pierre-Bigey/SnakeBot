from SnakeBot.genetic_algorithm.crossover import single_point_crossover
from SnakeBot.genetic_algorithm.mutation import mutate
from SnakeBot.genetic_algorithm.population import Population
from SnakeBot.genetic_algorithm.selection import tournament_selection


class GAController:
    def __init__(self, population_size, genome_length, fitness_function, grid_size):
        self.population = Population(population_size, genome_length, fitness_function, grid_size)
        self.generation = 0

    def evolve(self, generation):
        """Run the genetic algorithm for a given number of generations."""
        for _ in range(generation):
            # Evaluate the fitness of the population
            self.population.evaluate_fitness()

            # Create a new generation
            new_population = []

            for _ in range(self.population.population_size // 2):  # Create pairs of parents
                parent1 = tournament_selection(self.population.individuals)
                parent2 = tournament_selection(self.population.individuals)

                # Crossover to create new children
                child1 = single_point_crossover(parent1, parent2)
                child2 = single_point_crossover(parent2, parent1)

                # Mutate the children
                mutate(child1)
                mutate(child2)

                # Add new children to the population
                new_population.extend([child1, child2])

                # Replace the old population with the new one
            self.population.individuals = new_population
            self.generation += 1

            # Return the best individual after evolution
        return self.population.get_best_individual()
