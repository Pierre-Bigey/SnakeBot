import random

from SnakeBot.genetic_algorithm.individual import Individual


def single_point_crossover(parent1, parent2):
    """Perform a single-point crossover between two parents to create a child."""
    crossover_point = random.randint(1, len(parent1.genome) - 1)

    # Create a new child by combining parts of both parents
    child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]

    # Create a new individual with the child's genome
    individual = Individual(genome_length=len(child_genome))
    individual.force_genome(child_genome)
    return individual


