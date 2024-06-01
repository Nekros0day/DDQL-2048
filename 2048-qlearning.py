import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time

# Initialize the pygame
pygame.init()

# Constants
SIZE = WIDTH, HEIGHT = 400, 450
TILE_SIZE = WIDTH // 4
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
FONT_COLOR = (119, 110, 101)
FONT = pygame.font.Font(None, 50)
SCORE_FONT = pygame.font.Font(None, 36)
ANIMATION_SPEED = 0  # Speed of tile movement in pixels per frame
FPS = 60

# Initialize screen
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption('2048')

class Tile:
    def __init__(self, value, x, y):
        self.value = value
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y

    def set_position(self, x, y):
        self.target_x = x
        self.target_y = y

    def update_position(self):
        # Move towards the target position each frame
        if self.x < self.target_x:
            self.x = min(self.x + ANIMATION_SPEED, self.target_x)
        elif self.x > self.target_x:
            self.x = max(self.x - ANIMATION_SPEED, self.target_x)
        
        if self.y < self.target_y:
            self.y = min(self.y + ANIMATION_SPEED, self.target_y)
        elif self.y > self.target_y:
            self.y = max(self.y - ANIMATION_SPEED, self.target_y)

    def draw(self, surface):
        rect = pygame.Rect(self.x, self.y, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(surface, get_tile_color(self.value), rect)
        if self.value != 0:
            text = FONT.render(str(self.value), True, FONT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)

def initialize_game():
    board = [[None] * 4 for _ in range(4)]
    add_new_tile(board)
    add_new_tile(board)
    return board

def add_new_tile(board):
    empty_cells = [(i, j) for i in range(4) for j in range(4) if board[i][j] is None]
    if not empty_cells:
        return
    i, j = random.choice(empty_cells)
    value = 2 if random.random() < 0.9 else 4
    board[i][j] = Tile(value, j * TILE_SIZE, i * TILE_SIZE + 50)

def get_tile_color(value):
    if value in TILE_COLORS:
        return TILE_COLORS[value]
    return (60, 58, 50)

def draw_board(board, score):
    screen.fill(BACKGROUND_COLOR)
    # Draw score
    score_text = SCORE_FONT.render(f"Score: {score}", True, FONT_COLOR)
    screen.blit(score_text, (10, 10))
    
    for row in board:
        for tile in row:
            if tile is not None:
                tile.draw(screen)
    pygame.display.flip()

def move_left(row):
    new_row = [tile for tile in row if tile is not None]
    for i in range(len(new_row) - 1):
        if new_row[i] is not None and new_row[i + 1] is not None and new_row[i].value == new_row[i + 1].value:
            new_row[i].value *= 2
            new_row[i + 1] = None
    new_row = [tile for tile in new_row if tile is not None]
    while len(new_row) < 4:
        new_row.append(None)
    return new_row

def transpose(board):
    return [list(row) for row in zip(*board)]

def reverse(board):
    return [list(reversed(row)) for row in board]

def set_tile_positions(board):
    for i in range(4):
        for j in range(4):
            if board[i][j] is not None:
                board[i][j].set_position(j * TILE_SIZE, i * TILE_SIZE + 50)

def move(board, direction):
    if direction == 'UP':
        board = transpose(board)
        new_board = [move_left(row) for row in board]
        board = transpose(new_board)
    elif direction == 'DOWN':
        board = transpose(board)
        new_board = reverse([move_left(row) for row in reverse(board)])
        board = transpose(new_board)
    elif direction == 'LEFT':
        new_board = [move_left(row) for row in board]
        board = new_board
    elif direction == 'RIGHT':
        new_board = reverse([move_left(row) for row in reverse(board)])
        board = new_board
    set_tile_positions(board)
    return board

def boards_are_equal(board1, board2):
    return all((board1[i][j] is None and board2[i][j] is None) or 
               (board1[i][j] is not None and board2[i][j] is not None and board1[i][j].value == board2[i][j].value)
               for i in range(4) for j in range(4))

def game_over(board):
    for row in board:
        if any(tile is None for tile in row):
            return False
    for i in range(4):
        for j in range(3):
            if board[i][j] is not None and board[i][j + 1] is not None and board[i][j].value == board[i][j + 1].value:
                return False
            if board[j][i] is not None and board[j + 1][i] is not None and board[j][i].value == board[j + 1][i].value:
                return False
    return True



def calculate_score(board):
    return sum(tile.value for row in board for tile in row if tile is not None)

# Neural Network for DQN with Dueling Architecture
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Linear architecture: Assuming input state is flattened to 1D tensor of size 16 (4x4 grid)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output layers for value and advantage streams
        self.fc_value = nn.Linear(128, 1)
        self.fc_adv = nn.Linear(128, 4)  # Output for each action

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the state to a 1D tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.fc_value(x)
        adv = self.fc_adv(x)
        
        q_values = value + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN agent with Double DQN
class DQNAgent:
    def __init__(self):
        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(1000000)
        self.update_target_model()
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.best_score = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

def get_board_state(board):
    state = []
    for row in board:
        for tile in row:
            if tile is None:
                state.append(0)
            else:
                state.append(tile.value)
    return np.array(state).flatten()


def train():
    agent = DQNAgent()
    episodes = 10000

    for e in range(episodes):
        board = initialize_game()
        score = calculate_score(board)
        state = get_board_state(board)
        draw_board(board, score)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            previous_board = [row[:] for row in board]
            if action == 0:
                new_board = move(board, 'UP')
            elif action == 1:
                new_board = move(board, 'DOWN')
            elif action == 2:
                new_board = move(board, 'LEFT')
            elif action == 3:
                new_board = move(board, 'RIGHT')
            else:
                new_board = board

            if not boards_are_equal(previous_board, new_board):
                add_new_tile(new_board)
                board = new_board

            next_state = get_board_state(board)
            new_score = calculate_score(board)
            reward = new_score - score
            score = new_score

            done = game_over(board)
            
            for row in board:
                for tile in row:
                    if tile is not None:
                        tile.update_position()
                        
            # Update the screen during training
            draw_board(board, score)
            pygame.display.flip()

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent.update_target_model()
                if score > agent.best_score:
                    agent.best_score = score
                    agent.save("best_dqn_2048.pth")
                print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2}")
                break

            agent.replay()
           
            if e > episodes * 0.3:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    agent.save("dqn_2048.pth")
    
def play():
    agent = DQNAgent()
    agent.load("best_dqn_2048.pth")

    board = initialize_game()
    score = calculate_score(board)
    state = get_board_state(board)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                running = False

        previous_board = [row[:] for row in board]
        action = agent.act(state)

        if action == 0:
            new_board = move(board, 'UP')
        elif action == 1:
            new_board = move(board, 'DOWN')
        elif action == 2:
            new_board = move(board, 'LEFT')
        elif action == 3:
            new_board = move(board, 'RIGHT')
        else:
            new_board = board

        if not boards_are_equal(previous_board, new_board):
            add_new_tile(new_board)
            board = new_board
            score = calculate_score(board)

        state = get_board_state(board)

        # Update tile positions
        for row in board:
            for tile in row:
                if tile is not None:
                    tile.update_position()

        draw_board(board, score)
        pygame.display.flip()
        clock.tick(FPS)
        #time.sleep(0.1)

        if game_over(board):
            print("Game Over!")
            running = False

    pygame.quit()

def main():
    choice = input("Do you want to train (t) a new model or load (l) an existing model? ")
    if choice == 't':
        train()
    elif choice == 'l':
        play()
    else:
        print("Invalid choice. Please enter 't' or 'l'.")

if __name__ == "__main__":
    main()
