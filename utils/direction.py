from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def from_index(index):
        directions = list(Direction)
        if 0 <= index < len(directions):
            return directions[index]
        else:
            raise ValueError("Invalid index for direction")
