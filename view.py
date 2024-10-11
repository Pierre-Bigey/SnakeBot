"""Optional module for displaying the game state in a window."""
import tkinter as tk
from game import Game
from utils.config import CELL_SIZE

class WindowView:
    def __init__(self, game: Game):
        self.game = game
        self.window = tk.Tk()
        self.window.title("Snake Game")

        self.grid_size = self.game.grid_size
        self.canvas = tk.Canvas(self.window, width=(self.grid_size+1) * CELL_SIZE, height=(self.grid_size+1) * CELL_SIZE)
        self.canvas.pack()


    def render(self):
        """Render the game state in the window"""
        self.canvas.delete("all")

        #The snake head is in lime color

        x, y = self.game.snake.body[0]
        self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                     (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                     fill="lime")

        #Draw the snake body
        for x, y in self.game.snake.body[1:]:
            self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                         (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                         fill="green")

        #Draw the food
        x, y = self.game.food.position
        self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                     (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                     fill="red")

        self.window.update()


