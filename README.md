# 2048 with Deep Q-Learning (DQN)

This project implements the 2048 game using Pygame and integrates a Deep Q-Learning (DQN) agent to play the game. The agent is trained using a neural network with a dueling architecture.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Game Mechanics](#game-mechanics)
- [AI Agent](#ai-agent)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using pip:


## Usage

You can choose to either train a new model or load an existing model to play the game.

1. **Training a new model:**

   
When prompted, enter `t` to start training the model.

2. **Loading an existing model:**

   
When prompted, enter `l` to load the pre-trained model and watch the AI play the game.

## Game Mechanics

The game consists of a 4x4 grid of tiles. Each tile can be one of several values (2, 4, 8, 16, etc.). The goal is to combine tiles of the same value to create a tile with the value of 2048.

### Controls
- **UP:** Move all tiles up.
- **DOWN:** Move all tiles down.
- **LEFT:** Move all tiles left.
- **RIGHT:** Move all tiles right.

### Tile Colors
Each tile has a specific color based on its value for visual differentiation.

## AI Agent

The AI agent uses a Deep Q-Network (DQN) with a dueling architecture to learn the best moves in the game. It leverages a replay memory to store past experiences and train the network.

### Neural Network Architecture
- Two fully connected layers with ReLU activation.
- Separate streams for calculating the value and advantage functions.

### Training
- **Replay Memory:** Stores past experiences to train the network.
- **Double DQN:** Uses a target network to stabilize training.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation.

## Project Structure

- `main.py`: Main script to run the game and train/load the model.
- `dqn_agent.py`: Contains the DQN agent and neural network definition.
- `game.py`: Implements the game logic and rendering.
