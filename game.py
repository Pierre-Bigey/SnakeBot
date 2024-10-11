from snake import Snake, Direction
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


