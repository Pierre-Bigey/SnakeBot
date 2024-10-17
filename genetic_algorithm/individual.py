import random
from SnakeBot.utils.direction import Direction


class Individual:
    def __init__(self, genome_length=10):
        """Initialize the individual with a random genome"""
        self.genome_length = genome_length
        self.genome = [random.uniform(-1, 1) for _ in range(genome_length)]
        self.fitness = 0

    def force_genome(self, genome):
        """Force the individual to have a specific genome."""
        self.genome = genome

    def evaluate_fitness(self, game, fitness_function):
        """Run the game for this individual and compute its fitness."""

        while not game.is_over:
            # Use the bot's genome to play the game and compute the fitness
            bot_direction = self.decide_move(game.get_state())
            game.update(bot_direction)

        # Compute the fitness using the fitness function
        self.fitness = fitness_function(game)

    def decide_move(self, game_state):
        """Use the genome to decide the next move for the snake."""
        snake_head = game_state["snake_body"][0]
        food_position = game_state["food_position"]

        dx = food_position[0] - snake_head[0]
        dy = food_position[1] - snake_head[1]

        # Use the genome to decide the next move
        if self.genome[0] * dx + self.genome[1] * dy > 0:
            # Move toward the food horizontally
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            # Move toward the food vertically
            return Direction.DOWN if dy > 0 else Direction.UP
