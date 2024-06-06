import numpy as np
import os

from environment.scheduling_env import SchedulingEnv
from environment.model_apps import apps
from environment.model_infra import Infra

save_file_results = "data/results_custom.csv"


N_APPS = len(apps)
N_INFRA = Infra().getInfraSize()

class Scheduler:
    def __init__(self, type:str):     
        self.action = None
        if type == "random":
            def random_action():
                action = np.random.rand(N_INFRA)
                return action / np.sum(action)
            self.action = random_action
        
        elif type == "cloud":
            action = np.zeros(N_INFRA)
            action[0] = 1
            self.action = lambda : action 

        elif type == "edge":
            action = np.zeros(N_INFRA)
            for i in range(1, N_INFRA):
                action[i] = i / (N_INFRA - 1)
            action = action / np.sum(action)
            self.action = lambda : action
        else:
            raise ValueError("Invalid type of scheduler")
        
    def getAction(self):
        return {str(a) : self.action() for a in range(N_APPS)}


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    env = SchedulingEnv()
    scheduler = Scheduler("cloud")

    total_rewards = []
    total_obs = []
    total_infos = []
    total_actions = []
    total_energy = []
    
    obs,info = env.reset()
    for _ in range(1000):
        action = scheduler.getAction()
        obs, rewards, _,_, info = env.step(action)
        total_rewards.append(list(rewards.values()))
        total_obs.append(np.array(list(obs.values())))
        total_infos.append(np.array(info["appLoad"]))
        total_energy.append(info["energy"])
        total_actions.append(np.array(list(action.values())))

    with open(save_file_results, "w") as f:
        for (obs, actions, reward, info, energy) in zip(total_obs, total_actions,total_rewards,total_infos, total_energy):
            line =  ",".join(str(x) for x in obs.flatten()) + "," 
            line += ",".join(str(x) for x in actions.flatten()) + "," 
            line += ",".join(str(x) for x in info.flatten()) + ","
            line += str(energy) + ","
            line += ",".join(str(x) for x in reward) + "\n" 
            f.write(line)