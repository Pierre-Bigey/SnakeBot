import numpy as np

from genetic_algorithm.individual import Individual
from genetic_algorithm.mutation import mutate
from genetic_algorithm.population import Population

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def moyenne(liste):
    return sum(liste)/len(liste)

class GAController:
    def __init__(self, population_size, grid_size, hidden_size, fitness_function, game_iterations, generations):
        self.population = Population(population_size, grid_size, hidden_size, fitness_function, game_iterations )
        self.hidden_size = hidden_size
        self.game_iterations = game_iterations
        self.generations = generations

        self.x = np.array([])
        self.y = np.array([])

        # enable interactive mode
        plt.ion()

        # creating subplot and figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line1, = self.ax.plot(self.x, self.y)

        # setting labels
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.title("Updating plot...")



    def evolve(self):
        """Run the genetic algorithm for a given number of generations."""
        for i in range(self.generations):
            # Evaluate fitness of the current population
            self.population.evaluate_fitness()
            # Sort population by fitness
            self.population.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

            best_fit = round(self.population.individuals[0].fitness)
            #print("Generation ",i," best : ", best_fit)

            # Keep the top 25%
            top_25_percent = [ind.copy() for ind in self.population.individuals[:self.population.size // 4]]


            # If len> 10, then add the average of the last 10 values

            self.x = np.concatenate([self.x, [i]])
            self.y = np.concatenate([self.y, [moyenne([ind.fitness for ind in top_25_percent])]])


            self.line1.set_xdata(self.x)
            self.line1.set_ydata(self.y)

            #update xlin and ylim
            self.ax.relim()
            self.ax.autoscale_view()

            # re-drawing the figure
            self.fig.canvas.draw()

            # to flush the GUI events
            self.fig.canvas.flush_events()


            # Mutate the top 25% to create the next 50%
            next_50_percent = [ind.copy() for ind in top_25_percent] + [ind.copy() for ind in top_25_percent]
            for ind in next_50_percent:
                mutate(ind)

            # Create 50% random new individuals
            random_25_percent = [Individual(self.population.grid_size, self.hidden_size, self.game_iterations) for _ in range(self.population.size // 4)]

            # Form the new population
            self.population.individuals = top_25_percent + next_50_percent + random_25_percent


        # Return the best individual after evolution
        return top_25_percent[0]

