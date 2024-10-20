from view import View
from game import Game


def main():
    game = Game()
    view = View()

    game.snake = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    game.food = (9, 9)

    view.render(game)

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

    print("Snake: ", game.snake)
    print("Food: ", game.food)
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
