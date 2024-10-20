# GAME
GRID_SIZE = 10
ENERGY = 40
# DIRECTIONS
# UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
DIR = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# FITNESS
FITNESS_PER_FOOD = 1000
FITNESS_PER_ROUND = 1


# NEURAL NETWORK
INPUT_SIZE = 32
HIDDEN_SIZE_1 = 20
HIDDEN_SIZE_2 = 20
OUTPUT_SIZE = 4
SAVES_DIR = "saves"


# GENETIC ALGORITHM
POPUALTION_SIZE = 100
GENERATIONS = 1000
GAME_PER_SNAKE = GRID_SIZE * GRID_SIZE / 2
MUTATION_RATE = 0.05

BEST_PART = 0.25
RANDOM_PART = 0.5


# VIEW
CELL_SIZE = 20
SNAKE_COLOR = "green"
SNAKE_HEAD_COLOR = "lime"
FOOD_COLOR = "red"
FPS = 15

SHOW_TIME = False

