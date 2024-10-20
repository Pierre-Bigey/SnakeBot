import numpy as np
import random
from utils.config import *


class Game:
    def __init__(self):
         #Constant copy from utils.config to speed up the game
        self.grid_size = GRID_SIZE
        self.base_energy = ENERGY
         
        
        self.round = None
        self.food = None
        self.direction = None
        self.tail_direction = None
        self.alive = None
        self.snake = None
        self.energy = None
        self.reset()

    def reset(self):
        """Initialize or reset the game state."""

        # Set up the initial snake position (start in the middle of the grid).
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]  # Snake is a list of (row, col) tuples representing its body.

        # Place food at a random location.
        self.place_food()

        # Initialize other variables.
        self.direction = (0, 1)  # Initial direction: right (row change, col change)
        self.tail_direction = (0, 1)
        self.alive = True
        self.round = 0
        self.energy = self.base_energy

    def place_food(self):
        """Place food in a random empty location."""
        grid = np.zeros((self.grid_size, self.grid_size))

        for row, col in self.snake:
            grid[row, col] = 2
        empty_cells = np.argwhere(grid == 0)

        if len(empty_cells) > 0:
            food_position = random.choice(empty_cells)
            self.food = food_position

    def step(self, new_direction):
        """Advance the game by one step (move the snake)."""

        # new_direction is expected to be a tuple like (0, 1) for right, (-1, 0) for up, etc.
        # Ensure the new direction isn't the opposite of the current one.
        if new_direction != (-self.direction[0], -self.direction[1]):
            self.direction = new_direction
        else:
            self.alive = False
            return self.alive, self.round, len(self.snake)

        self.round += 1

        # Decrease energy by 1 in each step.
        self.energy -= 1
        if self.energy <= 0:
            self.alive = False

        # Check if the game is over.
        if not self.alive:
            return self.alive, self.round, len(self.snake)

        # Get the current head position of the snake.
        head_row, head_col = self.snake[0]

        # Calculate new head position based on the current direction.
        new_head_row = head_row + self.direction[0]
        new_head_col = head_col + self.direction[1]

        # Check if the snake hits the wall or itself.
        if (new_head_row < 0 or new_head_row >= self.grid_size or
                new_head_col < 0 or new_head_col >= self.grid_size or
                (new_head_row, new_head_col) in self.snake):
            self.alive = False
            return self.alive, self.round, len(self.snake)

        # Check if food is found.
        if new_head_row== self.food[0] and new_head_col == self.food[1]:
            self.snake = [(new_head_row, new_head_col)] + self.snake  # Grow the snake.
            self.place_food()  # Place new food.
            self.energy = self.base_energy  # Reset energy.
        else:
            # Move the snake: add the new head and remove the tail.
            self.snake = [(new_head_row, new_head_col)] + self.snake[:-1]

        # Update the tail direction.
        tail_row, tail_col = self.snake[-1]
        #if the snake has more than one part
        if len(self.snake) > 1:
            self.tail_direction = (self.snake[-2][0] - tail_row, self.snake[-2][1] - tail_col)
        else: #if the snake has only one part
            self.tail_direction = self.direction


        return self.alive, self.round, len(self.snake)

    def get_state(self):
        """Get the current game state (snake position, food position)."""
        return {
            'snake': self.snake,
            'food': self.food,
        }

    def get_snake_vision(self):
        """Calculate the snake's vision in 8 directions, including distance to wall, food, and body."""
        # Placeholder for the vision array (24 values in total).
        vision = np.zeros(24)

        # The vision array is divided in 8 parts, each part representing a direction.:
        # 7|0|1
        # 6|H|2
        # 5|4|3

        # The first part is the distance to the wall
        # The second part is a boolean if there is food in that direction
        # The third part is a boolean if there is a body part in that direction

        # Get the head position of the snake.
        head_row, head_col = self.snake[0]

        # Get all the wall distance (normalized)

        gr_2 = self.grid_size * 2

        vision[0] = head_row / self.grid_size  # North
        vision[2] = (self.grid_size - head_col) / self.grid_size  # East
        vision[4] = (self.grid_size - head_row) / self.grid_size  # South
        vision[6] = head_col / self.grid_size  # West

        vision[1] = 2 * min(vision[0], vision[2])  # North-East
        vision[3] = 2 * min(vision[2], vision[4])
        vision[5] = 2 * min(vision[4], vision[6])
        vision[7] = 2 * min(vision[6], vision[0])

        # Check if food is in each direction
        food_row, food_col = self.food
        food_check = [0, 0, 0, 0, 0, 0, 0, 0]
        #Check if the food is in the same row or column
        if food_row < head_row and food_col == head_col:
            food_check[0] = 1
        elif food_row == head_row and food_col > head_col:
            food_check[2] = 1
        elif food_row > head_row and food_col == head_col:
            food_check[4] = 1
        elif food_row == head_row and food_col < head_col:
            food_check[6] = 1

        elif food_row < head_row and food_col > head_col and (head_row - food_row) / (food_col - head_col) == 1:
            food_check[1] = 1
        elif food_row > head_row and food_col > head_col and (food_row - head_row) / (food_col - head_col) == 1:
            food_check[3] = 1
        elif food_row > head_row and food_col < head_col and (food_row - head_row) / (food_col - head_col) == 1:
            food_check[5] = 1
        elif food_row < head_row and food_col < head_col and (head_row - food_row) / (head_col - food_col) == 1:
            food_check[7] = 1

        vision[8:16] = food_check

        # For each direction, a value of 1 means that there is a body part in that direction.
        # Check each direction and check if a body part is there.
        body_check = [0, 0, 0, 0, 0, 0, 0, 0]
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        for direction in directions:
            line = [(head_row + i * direction[0], head_col + i * direction[1]) for i in range(1, self.grid_size)]
            for i in range(len(line)):
                if line[i] in self.snake:
                    body_check[directions.index(direction)] = 1
                    break

        vision[16:] = body_check

        return vision

    def get_input_layer(self):
        """Get the input layer for the neural network."""
        # The input layer is compose of the 24 values of the vision array
        # added with 2 * 4 hot encoded values for the direction of the snake head and direction of snake tail
        vision = self.get_snake_vision()
        tail_direction_encoded = [0, 0, 0, 0]
        tail_direction_encoded[DIR.index(self.tail_direction)] = 1
        head_direction_encoded = [0, 0, 0, 0]
        head_direction_encoded[DIR.index(self.direction)] = 1

        return np.concatenate((vision, head_direction_encoded, tail_direction_encoded))