import random

class Food:
    def __init__(self, grid_size, snake_body):
        self.grid_size = grid_size
        self.snake_body = snake_body
        self.position = self.generate_position()

    def generate_position(self):
        """Generate a random position for the food that is not occupied by the snake"""
        i = 0
        while True:
            i += 1
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)

            #As it's a while True loop, I check if the loop has run for more than 2 * grid_size ** 2 times
            #If it has, I return an error message
            if i > 2 * self.grid_size ** 2:
                raise Exception("The snake is too big for the grid, cannot find place for food")

            if (x, y) not in self.snake_body:
                return x, y