import numpy as np

from constants import TIME_PERIOD, N_APPS
from model_apps import apps

class App:
    def __init__(self, id, consumption_CPU = 0, consumption_BW = 0, consumption_run = 0, consumption_start = 0, distribution = lambda x: 0) -> None:
        self.id = id

        self.consumption_CPU = consumption_CPU # k instructions needed to execute a request
        self.consumption_BW = consumption_BW # kilo bytes needed to execute a request

        self.consumption_run = consumption_run # k instructions needed to run the service
        self.consumption_start = consumption_start # k instructions needed to start the service

        self.distribution = distribution

class PM:
    def __init__(self, id, apps:list[App], cpu = 0, bw = 0, consumption_idle = 0, consumption_max = 0) -> None:
        self.id = id

        self.CPU = cpu # max k instructions per second
        self.BW = bw # max kilo bytes per second

        self.CPU_load = 0 # current k instructions in period T
        self.BW_load = 0 # current kilo bytes in period T

        self.consumption_idle = consumption_idle # energy consumed when idle
        self.consumption_max = consumption_max # energy consumed when at max load

        self.apps = apps
        self.currentApps = [0]*len(apps)
        self.lastApps = [0]*len(apps)

    def powerUsage(self):
        if self.CPU_load == 0 and self.BW_load == 0:
            return 0
        
        energy = self.consumption_idle
        energy += (self.consumption_max-self.consumption_idle)*(0.8*(self.CPU_load/(T*self.CPU)) + 0.2*(self.BW_load/(T*self.BW)))

        for id, (app, old_app) in enumerate(zip(self.currentApps, self.lastApps)):
            if app != 0:
                energy += self.apps[id].consumption_run
                if old_app != 0:
                    energy += self.apps[id].consumption_start

        return energy
    
    def resetLoad(self):
        self.CPU_load = 0
        self.BW_load = 0
        
        self.usedApps = self.currentApps.copy()
        self.currentApps = [0]*len(self.apps)

    def addRequest(self, app: App):
        self.currentApps[app.id] += 1

        self.CPU_load += app.consumption_CPU
        self.BW_load += app.consumption_BW
        

class SBC(PM):
    def __init__(self, id, apps) -> None:
        super().__init__(id, apps, 1000000, 10000, 2, 4)

class Cloud(PM):
    def __init__(self, id, apps) -> None:
        super().__init__(id, apps, 1000000, 1000000, 100, 1000)


class Infra():
    def __init__(self) -> None:
        self.infra = [Cloud(0,apps)] + [SBC(i,apps) for i in range(1,5)]

    def getPowerUsage(self):
        return sum([pm.powerUsage() for pm in self.infra])
    
    def getQoS(self):
        return 0
    
    def getQoS_penalty(self):
        QoS_penalty = 0
        for pm in self.infra:
            if pm.CPU_load > pm.CPU*TIME_PERIOD:
                QoS_penalty += pm.CPU_load - pm.CPU*TIME_PERIOD
        return QoS_penalty
    
    def resetLoad(self):
        for pm in self.infra:
            pm.resetLoad()

    def addRequests(self, requests : np.ndarray[int], distribution : list[float]):
        if np.sum(requests) == 0: return 
        request = np.random.choice(range(len(N_APPS)), p=requests/np.sum(requests))
        requests[request] -= 1
        
        while np.sum(requests) > 0:
            pm_id = 0
            if np.sum(distribution[request]) == 0:
                pm_id = np.random.choice(range(len(self.infra)))
            else:
                pm_id = np.random.choice(range(len(self.infra)), p=distribution[request])
            self.infra[pm_id].addRequest(self.apps[request])

            request = np.random.choice(range(len(N_APPS)), p=requests/np.sum(requests))
            requests[request] -= 1

            