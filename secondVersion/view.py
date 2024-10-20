
"""Class that contains the view of the application.
The view will take a game state and render it to the screen."""

import tkinter as tk
import utils.config as config

class View:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Snake')
        self.root.resizable(False, False)

        self.cell_size = config.cell_size
        self.snake_color = config.snake_color
        self.snake_head_color = config.snake_head_color
        self.food_color = config.food_color
        self.grid_size = config.grid_size

        self.canvas = tk.Canvas(self.root, width= (self.grid_size + 2) * self.cell_size, height= (self.grid_size + 2) * self.cell_size)


        self.canvas.pack()

    def render(self, game):
        """Render the game state in the window"""
        self.canvas.delete('all')

        # Draw the grid
        for i in range(1,self.grid_size + 2):
            self.canvas.create_line(i * self.cell_size, self.cell_size, i * self.cell_size, (self.grid_size+1) * self.cell_size)
            self.canvas.create_line(self.cell_size, i * self.cell_size, (self.grid_size+1) * self.cell_size, i * self.cell_size)

        # Draw the snake head
        y, x = game.snake[0]
        x += 1
        y += 1
        self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                     (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                     fill=self.snake_head_color)

        # Draw the snake body
        for y, x in game.snake[1:]:
            x += 1
            y += 1
            self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                         (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                         fill=self.snake_color)

        # Draw the food
        y, x = game.food
        x += 1
        y += 1
        self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                     (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                     fill=self.food_color)

        self.root.update()
