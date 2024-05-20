import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

from constants import *
from model_apps import getRequests
from model_infra import Infra


class CustomEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "scheduling_environment_v0",
    }

    def __init__(self):
        self.requests : list[int] = []
        self.nextRequests : list[int] = []
        self.requestsHistory : list[np.ndarray[int]] = []
        self.agents = list(range(N_APPS))
        self.infra = Infra()
        self.timestep = 0

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.timestep = random.randint(0, 100)
        self.infra.resetLoad()

        observations = {
            a: () for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        self.timestep += 1

        # Execute actions
        distribution = np.array([actions[a] for a in self.agents])

        self.requests = getRequests(self.timestep)

        self.infra.resetLoad()
        self.infra.addRequests(self.requests, distribution)

        reward = (
            -self.infra.getPowerUsage() 
            - LAMBDA*self.infra.getQoS()
            - np.exp(LAMBDA_2*self.infra.getQoS_penalty())) 

        rewards = {a: reward for a in self.agents}   
        
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        
        # Check truncation conditions (overwrites termination conditions) TODO
        truncations = {a: False for a in self.agents}

        # Get observations TODO
        state = ()
        state = state + self.infra.getLoadCPU()
        state = state + self.infra.getLoadBW()

        observations = {
            a: (
                state + self.requests[a]
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        grid = np.full((7, 7), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(2*N_PM + N_APPS + N_APPS*N_PM)

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_APPS)