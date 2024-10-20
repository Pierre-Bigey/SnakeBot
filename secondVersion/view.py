
"""Class that contains the view of the application.
The view will take a game state and render it to the screen."""

import tkinter as tk
from utils.config import *

class View:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Snake')


        self.canvas = tk.Canvas(self.root, width= (GRID_SIZE + 2) * CELL_SIZE, height= (GRID_SIZE + 2) * CELL_SIZE)
        self.canvas.pack()




    def render(self, game):
        """Render the game state in the window"""
        self.canvas.delete('all')

        # Draw the grid
        for i in range(1,GRID_SIZE + 2):
            self.canvas.create_line(i * CELL_SIZE, CELL_SIZE, i * CELL_SIZE, (GRID_SIZE+1) * CELL_SIZE)
            self.canvas.create_line(CELL_SIZE, i * CELL_SIZE, (GRID_SIZE+1) * CELL_SIZE, i * CELL_SIZE)

        # Draw the snake head
        y, x = game.snake[0]
        x += 1
        y += 1
        self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                     (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                     fill=SNAKE_HEAD_COLOR)

        # Draw the snake body
        for y, x in game.snake[1:]:
            x += 1
            y += 1
            self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                         (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                         fill=SNAKE_COLOR)

        # Draw the food
        y, x = game.food
        x += 1
        y += 1
        self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                     (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                     fill=FOOD_COLOR)

        # Add a text to show the score
        self.canvas.create_text(CELL_SIZE * (GRID_SIZE + 1) // 2, CELL_SIZE // 2,
                                text=f"Score: {len(game.snake)-1}",
                                font=("Helvetica", int(CELL_SIZE /2) ),
                                fill="black")


        self.root.update()

    def close(self):
        self.root.destroy()

    def print_snake_vision(self,game):

        vision = game.get_snake_vision()

        distance = vision[:8]
        food = vision[8:16]
        body = vision[16:]

        print("Wall Distance")
        print(str(round(distance[7], 2)) + "|" + str(round(distance[0], 2)) + "|" + str(round(distance[1], 2)))
        print("-----------------")
        print(str(round(distance[6], 2)) + "| H |" + str(round(distance[2], 2)))
        print("-----------------")
        print(str(round(distance[5], 2)) + "|" + str(round(distance[4], 2)) + "|" + str(round(distance[3], 2)))

        print("\n\nFood")
        print(int(food[7]), "|", int(food[0]), "|", int(food[1]))
        print("---------")
        print(int(food[6]), "| H |", int(food[2]))
        print("---------")
        print(int(food[5]), "|", int(food[4]), "|", int(food[3]))

        print("\n\nBody")
        print(int(body[7]), "|", int(body[0]), "|", int(body[1]))
        print("---------")
        print(int(body[6]), "| H |", int(body[2]))
        print("---------")
        print(int(body[5]), "|", int(body[4]), "|", int(body[3]))