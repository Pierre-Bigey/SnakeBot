from utils.direction import Direction


class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # The snake starts with a length of 3 units
        self.body = [(grid_size // 2, grid_size // 2), (grid_size // 2 - 1, grid_size // 2),
                     (grid_size // 2 - 2, grid_size // 2)]
        # The snake starts moving to the right
        self.direction = Direction.RIGHT
        self.grow_next = False

    def change_direction(self, new_direction: Direction):
        """Change the direction of the snake"""
        # Prevent the snake from going in the opposite direction
        if (new_direction.value[0] * -1, new_direction.value[1] * -1) == self.direction.value:
            return
        self.direction = new_direction

    def move(self):
        # Move the snake in the current direction
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction.value[0], head_y + self.direction.value[1])

        # Add new head to the body
        self.body = [new_head] + self.body

        # If the snake is not growing, remove the tail
        if not self.grow_next:
            self.body.pop()
        else:
            self.grow_next = False

    def grow(self):
        # Grow the snake by one unit
        self.grow_next = True

    def has_collided(self):
        # Check if the snake has collided with the walls or itself
        head_x, head_y = self.body[0]

        # Check if the snake hits the walls
        if head_x < 0 or head_x >= self.grid_size or head_y < 0 or head_y >= self.grid_size:
            return True

        # Check if the snake hits itself
        if (head_x, head_y) in self.body[1:]:
            return True

        return False
