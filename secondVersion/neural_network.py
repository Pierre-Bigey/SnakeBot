import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.config import *


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1)  # Input layer (32 inputs -> 20 hidden neurons)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)  # Hidden layer (20 neurons -> 12 neurons)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE)  # Output layer (12 neurons -> 4 output directions)

    def forward(self, x):
        # Pass the input through the first hidden layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Pass through the second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))

        # Output layer with softmax to get direction probabilities
        x = F.softmax(self.fc3(x), dim=1)

        return x


def choose_direction(model, game):
    """
    Use the neural network to choose a direction based on the game state.
    :param model: The neural network model (SnakeNet)
    :param game: The Game object
    :return: The direction tuple (one of (-1, 0), (1, 0), (0, -1), (0, 1))
    """
    # Get the input layer (vision + encoded direction info)
    inputs = game.get_input_layer()

    # Convert the input to a PyTorch tensor and add batch dimension
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 32)

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Perform a forward pass through the network
        output = model(inputs)

    # Choose the direction with the highest probability
    direction_idx = torch.argmax(output).item()

    # Convert the index back to a direction tuple (up, down, left, right)
    return DIR[direction_idx]


def save_best_model(model, filename="best_01.pth"):
    """
    Save the best-performing model to a file.
    :param model: The model to be saved (PyTorch model).
    :param filename: The file path to save the model.
    """


    if not os.path.exists(SAVES_DIR):
        os.makedirs(SAVES_DIR)

    torch.save(model.state_dict(), SAVES_DIR+"/"+filename)
    print(f"Best model saved to : "+SAVES_DIR+"/"+filename)


def load_best_model(filename="best_01.pth"):
    """
    Load the best-performing model from a file.
    :param filename: The file path to load the model from.
    :return: The loaded model (PyTorch model).
    """

    if not os.path.exists(SAVES_DIR):
        raise FileNotFoundError(f"The directory {SAVES_DIR} does not exist.")

    file_path = os.path.join(SAVES_DIR, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    model = SnakeNet()
    model.load_state_dict(torch.load(SAVES_DIR+"/"+filename))
    return model
