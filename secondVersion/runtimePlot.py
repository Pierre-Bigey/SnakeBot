import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from utils.config import SHOW_TIME

import time

matplotlib.use('TkAgg')

class Plot:
    def __init__(self ):
        self.generation = np.array([])
        self.fitness = np.array([])
        self.mvn_avg = np.array([])

        self.time_passed = np.array([])

        # enable interactive mode
        plt.ion()

        # creating subplot and figure
        if SHOW_TIME:
            self.fig1, (self.ax1, self.ax2) = plt.subplots(2, 1)  # Create two subplots
        else:
            self.fig1, self.ax1 = plt.subplots(1, 1)

        self.ax1.set_title("Fitness over generations")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness")
        self.avg_fitness_line, = self.ax1.plot(self.generation, self.fitness, 'b-', label="Best fitness")
        self.mean_fitness_line, = self.ax1.plot(self.generation, self.mvn_avg, 'r-', label="Moving average")

        #Add legend to the plot
        self.ax1.legend()

        if SHOW_TIME:
            self.ax2.set_title("Time passed")
            self.ax2.set_xlabel("Generation")
            self.ax2.set_ylabel("Time taken (s)")

            self.time_line, = self.ax2.plot(self.generation, self.time_passed, 'g-')

            self.last_time = time.time()

        self.exit_requested = False
        # Bind the close event to handle Tkinter window close gracefully
        self.fig1.canvas.mpl_connect('close_event', self.on_close)

    def add_fitness(self, best_fitness):
        self.generation = np.concatenate([self.generation, [len(self.generation)+1]])
        self.fitness = np.concatenate([self.fitness, [best_fitness]])
        new_avg = np.mean(self.fitness[-10:])
        self.mvn_avg = np.concatenate([self.mvn_avg, [new_avg]])

        self.avg_fitness_line.set_xdata(self.generation)
        self.avg_fitness_line.set_ydata(self.fitness)

        self.mean_fitness_line.set_xdata(self.generation)
        self.mean_fitness_line.set_ydata(self.mvn_avg)

        # update xlin and ylim
        self.ax1.relim()
        self.ax1.autoscale_view()


        if SHOW_TIME:
            #Compute and print time
            time_passed = time.time() - self.last_time
            self.last_time = time.time()
            self.time_passed = np.concatenate([self.time_passed, [time_passed]])

            self.time_line.set_xdata(self.generation)
            self.time_line.set_ydata(self.time_passed)

            self.ax2.relim()
            self.ax2.autoscale_view()

        # re-drawing the figure
        self.fig1.canvas.draw()

        # to flush the GUI events
        self.fig1.canvas.flush_events()

        return self.exit_requested

    def on_close(self, event):
        """Handle the window close event (set the exit flag)."""
        print("Window close detected. Requesting graceful exit...")
        self.exit_requested = True