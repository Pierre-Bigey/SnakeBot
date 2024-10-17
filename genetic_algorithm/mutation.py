import random


def mutate(individual, mutation_rate = 0.07):
    """Mutate an individual's genome with a given probability."""
    for i in range(len(individual.genome)):
        if random.random() < mutation_rate:
            individual.genome[i] += random.uniform(-0.1, 0.1)
