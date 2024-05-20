import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Box

from pettingzoo import ParallelEnv

import os
os.chdir(os.path.dirname(__file__))
print(os.getcwd())

from environment.constants import *
from environment.model_apps import getRequests, apps
from environment.model_infra import Infra

N_APPS = len(apps)
N_INFRA = Infra().getInfraSize()

class SchedulingEnv(ParallelEnv):
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
        self.infra = Infra()
        self.possible_agents = list(map(str,range(N_APPS)))
        self.timestep = 0

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.
        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = random.randint(0, 100)
        self.infra.resetLoad()
        self.requests = getRequests(self.timestep)

        # Reset observations 
        state = ()
        state = state + self.infra.getLoadCPU()
        state = state + self.infra.getLoadBW()

        observations = {
            a: (
                state + self.requests[a]
            )
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).
        And any internal state used by observe() or render()
        """
        self.timestep += 1

        # Execute actions
        distribution = np.array([actions[a] for a in self.agents])

        self.requests = getRequests(self.timestep)

        self.infra.resetLoad()
        self.infra.addRequests(self.requests, distribution)

        reward = (
            - self.infra.getPowerUsage() 
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
        print("====================================")
        print(f"Time: {self.timestep}")
        print("Requests: ", self.requests)
        print("Infra: ", self.infra.getLoadCPU())
        print("====================================")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0,high=2**60,shape=(2*N_INFRA + N_APPS + N_APPS*N_INFRA,),dtype=int)

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=0, high=1, shape=(N_INFRA,),dtype=float)