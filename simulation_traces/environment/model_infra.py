import numpy as np

from environment.constants import TIME_PERIOD, ENERGY_REQUEST_RATIO
from environment.model_apps import apps


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

    
    def startupCost(self):
        for id, (app, old_app) in enumerate(zip(self.currentApps, self.lastApps)):
            if app != 0:
                self.CPU_load += self.apps[id]["consumption_run"]
                if old_app != 0:
                    self.CPU_load += self.apps[id]["consumption_start"]

        self.lastApps = self.currentApps.copy()

    
    def powerUsage(self):
        if self.CPU_load == 0 and self.BW_load == 0:
            return 0
        
        energy = self.consumption_idle
        energy += (self.consumption_max-self.consumption_idle)*(0.8*(self.CPU_load/(TIME_PERIOD*self.CPU)) + 0.2*(self.BW_load/(TIME_PERIOD*self.BW)))

        return energy
    
    def powerUsageNormalized(self, totalRequests):
        if self.CPU_load == 0 and self.BW_load == 0:
            return 0
        
        energy = (self.consumption_max-self.consumption_idle)*(0.8*(self.CPU_load/(TIME_PERIOD*self.CPU)) + 0.2*(self.BW_load/(TIME_PERIOD*self.BW)))
        energy /= totalRequests
        energy *= ENERGY_REQUEST_RATIO
        energy += self.consumption_idle

        return energy
    
    def resetLoad(self):
        self.CPU_load = 0
        self.BW_load = 0
        
        self.usedApps = self.currentApps.copy()
        self.currentApps = [0]*len(self.apps)

    def addRequest(self, app: App):
        self.currentApps[app["id"]] += 1

        self.CPU_load += app["consumption_CPU"]
        self.BW_load += app["consumption_BW"]
    
    def addRequests(self, app: App, requests: int):
        self.currentApps[app["id"]] += requests

        self.CPU_load += app["consumption_CPU"]*requests
        self.BW_load += app["consumption_BW"]*requests
        

class SBC(PM):
    def __init__(self, id, apps) -> None:
        super().__init__(id, apps, 1000000, 10000, 2, 4)

class Cloud(PM):
    def __init__(self, id, apps) -> None:
        super().__init__(id, apps, 10000000, 10000000, 100, 1000)


class Infra():
    def __init__(self) -> None:
        self._infra : list[PM] = [SBC(i,apps) for i in range(2)]

    def getInfraSize(self) -> int:
        return len(self._infra)
    
    def getPowerUsage(self):
        return sum([pm.powerUsage() for pm in self._infra])
    
    def getNormalizedPowerUsage(self, totalRequests):
        return np.mean([pm.powerUsageNormalized(totalRequests)/pm.consumption_max for pm in self._infra])
    
    def getMaxPower(self):
        return sum([pm.consumption_max for pm in self._infra])
    
    def getQoS(self):
        return 0
    
    def getLoadCPU(self):
        return [pm.CPU_load/pm.CPU for pm in self._infra]
    
    def getLoadBW(self):
        return [pm.BW_load/pm.BW for pm in self._infra]
    
    def getAppReward(self, appId: int):
        app = apps[appId]
        wasted_CPU = 0
        for pm in self._infra:
            if pm.currentApps[appId]>0:
                wasted_CPU += min(1,0.9 - (pm.CPU_load) / (pm.CPU*TIME_PERIOD))

        return 1 - wasted_CPU/self.getInfraSize()

    
    def getQoS_penalty(self):
        QoS_penalty = 0
        for pm in self._infra:
            if pm.CPU_load > pm.CPU*TIME_PERIOD:
                QoS_penalty += pm.CPU_load - pm.CPU*TIME_PERIOD
        return QoS_penalty
    
    def resetLoad(self):
        for pm in self._infra:
            pm.resetLoad()

    def addRequests(self, requests : np.ndarray[int], distribution : list[list[float]], fast=True):
        """
        Adds requests to the infrastructure
        Params:
            requests : np.ndarray[int] : number of requests for each app
            distribution : list[list[float]] : distribution of requests for each app over the infrastructure
            fast : bool : if False, the requests are added one by one, otherwise all at once"""
        
        distribution = np.array(distribution)
        distribution -= 0.2 # TODO hyperparameter
        distribution[distribution < 0] = 0
        for i in range(len(distribution)):
            if np.sum(distribution[i]) <= 0:
                distribution[i] = np.ones(len(distribution[i]))
        distribution /= np.sum(distribution, axis=1)[:,None]
        
        if not fast:
            if np.sum(requests) == 0: return 
            
            requests_c = requests.copy()
            request = np.random.choice(range(len(apps)), p=requests_c/np.sum(requests_c))
            requests_c[request] -= 1
            
            while np.sum(requests_c) > 0:
                pm_id = 0
                if np.sum(distribution[request]) == 0:
                    pm_id = np.random.choice(range(self.getInfraSize()))
                else:
                    pm_id = np.random.choice(range(self.getInfraSize()), p=distribution[request]/np.sum(distribution[request]))
                self._infra[pm_id].addRequest(apps[request])

                request = np.random.choice(range(len(apps)), p=requests_c/np.sum(requests_c))
                requests_c[request] -= 1
        
        else:
            distribution_int = distribution[:,:]
            for i,requests in enumerate(requests):
                distribution_int[i] *= requests
            distribution_int = np.rint(distribution_int).astype(int)
            
            for app in range(len(distribution_int)):
                for pm in range(len(distribution_int[app])):
                    self._infra[pm].addRequests(apps[app], distribution_int[app][pm])

        for pm in self._infra:
            pm.startupCost()


    def addRequestsPriority(self, requests : np.ndarray[int], priority : list[list[float]], fast=False):
        """
        Adds requests to the infrastructure based on the priority of the apps
        Params:
            requests : np.ndarray[int] : number of requests for each app
            priority : list[list[float]] : priority of the PMs for each app
            fast : bool : if False, the requests are added one by one, otherwise all at once"""

        requests_c = requests.copy()
        request = np.random.choice(range(len(apps)), p=requests_c/np.sum(requests_c))
        requests_c[request] -= 1
        
        while np.sum(requests_c) > 0:
            pm_id = -1
            while pm_id == -1:
                pm_id = np.argmax(priority[request])
                if self._infra[pm_id].CPU_load > 0.9*self._infra[pm_id].CPU*TIME_PERIOD:
                    priority[request][pm_id] = 0
                    pm_id = -1
                if np.sum(priority) == 0:
                    pm_id = np.random.randint(self.getInfraSize())
            
            self._infra[pm_id].addRequest(apps[request])

            request = np.random.choice(range(len(apps)), p=requests_c/np.sum(requests_c))
            requests_c[request] -= 1

        for pm in self._infra:
            pm.startupCost()
