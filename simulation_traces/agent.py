import torch
import random
import numpy as np
from collections import deque
from simulation import Simulation
from constants import *
from model import QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = QNet(6*N_PM+5*N_APPS+N_APPS*N_PM, 512, N_APPS*N_PM)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, simulation: Simulation):
        return np.array(simulation.getState())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state)) # popleft if MAX_MEMORY is reached

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - self.n_games # TODO change epsilon
        final_move = np.zeros(N_APPS*N_PM)
        
        if random.randint(0, 200) < self.epsilon:
            for i in range(N_APPS):
                sum = 0
                for j in range(N_PM):
                    if np.random.rand() < 0.5:
                        final_move[i*N_PM+j] = 1
                        c += 1
                if sum == 0:
                    continue
                for j in range(N_PM):
                    final_move[i*N_PM+j] /= sum
                    
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    agent = Agent()
    simulation = Simulation()
    while True:
        # get old state
        state_old = agent.get_state(simulation)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward = simulation.tick(final_move) # TODO get more data to plot
        state_new = agent.get_state(simulation)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new)

        # remember
        agent.remember(state_old, final_move, reward, state_new)


if __name__ == '__main__':
    train()