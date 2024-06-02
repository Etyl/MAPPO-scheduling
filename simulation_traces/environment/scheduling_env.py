import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Box

from pettingzoo import ParallelEnv

from environment.constants import *
from environment.model_apps import getRequests, apps
from environment.model_infra import Infra

N_APPS = len(apps)
N_INFRA = Infra().getInfraSize()

MAX_SEARCH_REQUESTS = 200

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
        self.last_actions = None

        max_requests = np.zeros(N_APPS)
        for i in range(MAX_SEARCH_REQUESTS):
            max_requests = np.maximum(max_requests, getRequests(i))
        self.max_requests = max_requests

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.
        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = random.randint(0, 100)
        self.infra.resetLoad()
        self.requests = getRequests(self.timestep)

        # Reset observations 
        state = []
        state = state + self.infra.getLoadCPU()
        state = state + self.infra.getLoadBW()

        observations = {
            a: (
                tuple(state + [0]*N_INFRA + [self.requests[int(a)]/self.max_requests[int(a)]])
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
        self.infra.addRequestsPriority(self.requests, distribution)

        # Reward calculation

        power = self.infra.getNormalizedPowerUsage(np.sum(self.requests))
        qos = self.infra.getQoS() / TIME_PERIOD
        qos_penalty = self.infra.getQoS_penalty() / TIME_PERIOD
        
        rewardGlobal = (1.)*(1-power) + (0)*(1-qos) + (0)*(1-qos_penalty)

        divergenceReward = [0.5]*len(self.agents)
        # reward for divergence from last action
        if self.last_actions is not None:
            divergenceReward = [1-np.max(np.abs((actions[a] - self.last_actions[a]))) for a in self.agents]
        
        separationReward = [0.5]*len(self.agents)
        for k,a in enumerate(self.agents):
            sorted_actions = sorted(actions[a])
            S = 1
            for i in range(len(self.agents)-1):
                if sorted_actions[i+1] > 0.05:
                    S = min(S,abs(sorted_actions[i+1]-sorted_actions[i]))
            separationReward[k] = S

        rewards = {a: (.7)*rewardGlobal + (0.3)*divergenceReward[i] for (i,a) in enumerate(self.agents)}
 
        
        # reward from utilization of infrastructure
        # rewards = {a: self.infra.getAppReward(int(a)) for a in self.agents} 


        # Check termination conditions
        terminations = {a: False for a in self.agents}
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}

        # Reset observations 
        state = []
        state = state + self.infra.getLoadCPU()
        state = state + self.infra.getLoadBW()

        if self.last_actions is not None:
            observations = {
                a: (
                    tuple(state + list(actions[a]) + [self.requests[int(a)]/self.max_requests[int(a)]])
                )
                for a in self.agents
            }
        else:
            observations = {
                a: (
                    tuple(state + [0]*N_INFRA + [self.requests[int(a)]/self.max_requests[int(a)]])
                )
                for a in self.agents
            }

        # Get dummy infos (not used in this example)
        infos = {}
        infos["appLoad"] = self.infra.getAppLoad()

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        self.last_actions = actions

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
        return Box(low=0,high=1,shape=(3*N_INFRA + 1,),dtype=float)

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=0, high=1, shape=(N_INFRA,),dtype=float)