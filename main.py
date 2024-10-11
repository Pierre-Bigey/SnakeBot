from view import WindowView
from game import Game
from snake import Direction
from utils.config import FPS
import time
import random

def random_bot(game_state):
    return Direction(random.choice(list(Direction)))

def main(render= False, grid_size= 20, bot_control = True):
    game = Game(grid_size)
    view = WindowView(game) if render else None

    while not game.is_over:
        if bot_control:
            bot_direction = random_bot(game.get_state())
            game.update(bot_direction)

        else:
            game.update()

        if render:
            view.render()
            time.sleep(1/FPS)

    print("Game over! Score:", game.score)


if __name__ == "__main__":
    main(render=True, grid_size=20)