import numpy as np

from snake import Snake
from food import Food



class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake = Snake(grid_size)
        self.food = Food(grid_size, self.snake.body)
        self.score = 0
        self.round = 0
        self.is_over = False

    def update(self, bot_direction = None):
        """Update the game state : move snake, check for collisions, etc."""

        self.round += 1

        #Stop the game if the snake has taken more than 100 step to eat a food
        if self.round >= 100 * (len(self.snake.body)-2):
            self.is_over = True

        if bot_direction:
            self.snake.change_direction(bot_direction)

        self.snake.move()

        #Check if the snake has eaten the food
        if self.snake.body[0] == self.food.position:
            self.snake.grow()
            self.score += 1
            self.food = Food(self.grid_size, self.snake.body)

        #Check for collisions
        if self.snake.has_collided():
            self.is_over = True

    def get_state(self):
        """Return the current state of the game, for AI or rendering"""
        return {
            "snake_body": self.snake.body,
            "food_position": self.food.position,
            "score": self.score,
            "round": self.round,
            "game_over": self.is_over
        }

    def get_state_matrix(self):
        """Convert the game state into a matrix with 0 (empty), 1 (head), -50 (body), 25 (food)."""
        state_matrix = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Add the snake's body to the matrix (use -50 for body)
        for segment in self.snake.body[1:]:  # Skip the head
            state_matrix[segment[1]][segment[0]] = -50

        # Add the snake's head to the matrix (use 10 for head)
        head_x, head_y = self.snake.body[0]
        state_matrix[head_y][head_x] = 1

        # Add the food to the matrix (use 100 for food)
        food_x, food_y = self.food.position
        state_matrix[food_y][food_x] = 25

        return np.array(state_matrix) / 50  # Normalize the matrix to be between -1 and 1


