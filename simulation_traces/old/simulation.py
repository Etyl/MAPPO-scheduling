from trace_generator import TraceGenerator
from math import *
from infra import PM, App, SBC, Cloud
import numpy as np
from simulation_traces.old.predictor import Predictor
from constants import *


SCALE = 200
LAMBDA = 0.1 # TODO find a good value
LAMBDA_2 = 0.001 # TODO find a good value


class Data:
    def __init__(self):
        self.requests : list[int] = None
        self.pm_requests : list[int] = None
        self.pm_apps : list[list[int]] = None

class Simulation:
    def __init__(self) -> None:
        self.predictor : Predictor = Predictor()
        self.traceGenerator : TraceGenerator = TraceGenerator()
        self.requests : list[int] = []
        self.nextRequests : list[int] = []
        self.requestsHistory : list[np.ndarray[int]] = []
        self.apps : list[App] = []
    
        serverApp = App(0, 1000, 10, 0, 0)
        serverApp.distribution = lambda x: int(SCALE*sin(x/(2*pi*10))+1.2*SCALE)
        self.apps.append(serverApp)

        serverApp = App(1, 1000, 10, 0, 0)
        serverApp.distribution = lambda x: int(0.6*SCALE*sin(x/(2*pi*10)+140)+SCALE)
        self.apps.append(serverApp)

        serverApp = App(2, 1000, 10, 0, 0)
        serverApp.distribution = lambda x: int(0.3*SCALE*sin(x/(2*pi*10)+180)+0.3*SCALE)
        self.apps.append(serverApp)

        self.PMs = [Cloud(0,self.apps)] + [SBC(i,self.apps) for i in range(1,5)]


    def predictRequests(self) -> None:
        formatRequests = np.zeros(N_APPS)
        for request in self.requests:
            formatRequests[request] += 1
        self.requestsHistory.append(formatRequests)
        self.nextRequests = self.predictor.predict(self.requestsHistory)
        
    
    def getState(self) -> list[float]:
        """
        Can only be called once per tick (or change predictRequests)
        """
        self.predictRequests()
        state = []
        for pm in self.PMs:
            state.append(pm.CPU)
            state.append(int(pm.CPU_load/T))
            state.append(pm.BW)
            state.append(int(pm.BW_load/T))
            state.append(pm.consumption_idle)
            state.append(pm.consumption_max)

        for app in self.apps:
            state.append(app.consumption_CPU)
            state.append(app.consumption_BW)
            state.append(self.nextRequests[app.id])
            state.append(app.consumption_run)
            state.append(app.consumption_start)
            for pm in self.PMs:
                state.append(pm.currentApps[app.id])

        return state


    def tick(self, action : list[float]):
        
        self.traceGenerator.generate(self.apps)
        self.requestsHistory.append(self.traceGenerator.requests)
        action = np.array(action).reshape((N_APPS, N_PM))

        for pm in self.PMs:
            pm.resetLoad()
        request = self.traceGenerator.getRequest()
        while request is not None:
            pm_id = 0
            if np.sum(action[request]) == 0:
                pm_id = np.random.choice(range(len(self.PMs)))
            else:
                pm_id = np.random.choice(range(len(self.PMs)), p=action[request])
            self.PMs[pm_id].addRequest(self.apps[request])

            request = self.traceGenerator.getRequest()
        
        energy_cost = 0
        for pm in self.PMs:
            energy_cost += pm.powerUsage()
        
        QoS = 0
        # TODO get QoS
        
        # apply penalty for QoS if CPU load is higher than CPU capacity
        QoS_penalty = 0
        for pm in self.PMs:
            if pm.CPU_load > pm.CPU*T:
                QoS_penalty += pm.CPU_load - pm.CPU*T

        reward = -energy_cost - LAMBDA*QoS - np.exp(LAMBDA_2*QoS_penalty)

        data = Data()
        data.requests = self.requestsHistory[-1]
        data.pm_requests = [pm.CPU_load for pm in self.PMs]
        data.pm_apps = [pm.currentApps for pm in self.PMs]

        return reward, data
    
