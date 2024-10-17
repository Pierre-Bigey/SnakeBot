import random


def tournament_selection(population, tournament_size=3):
    """Select an individual from the population using tournament selection."""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)
