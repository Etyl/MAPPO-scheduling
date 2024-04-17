from trace_generator import TraceGenerator
from math import *
from infra import PM, App
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


SCALE = 20
LAMBDA = 0.1 # TODO find a good value


class Simulation:

    def __init__(self) -> None:
        self.traceGenerator : TraceGenerator = TraceGenerator()
        self.requests : list[int] = []
        self.PMs = [PM(i) for i in range(4)]
        self.apps : list[App] = []
    
        serverApp = App(0, 100, 10, 500, 500)
        serverApp.distribution = lambda x: int(SCALE*sin(x/(2*pi*10))+1.2*SCALE)
        self.apps.append(serverApp)

        serverApp = App(1, 100, 10, 500, 500)
        serverApp.distribution = lambda x: int(0.6*SCALE*sin(x/(2*pi*10)+140)+SCALE)
        self.apps.append(serverApp)

        serverApp = App(2, 100, 10, 500, 500)
        serverApp.distribution = lambda x: int(0.3*SCALE*sin(x/(2*pi*10)+180)+0.3*SCALE)
        self.apps.append(serverApp)


    def getState(self) -> list[float]:
        state = []
        for pm in self.PMs:
            state.append(pm.CPU_load)
            state.append(pm.BW_load)

        for app in self.apps:
            state.append(app.consumption_CPU)
            state.append(app.consumption_BW)
            state.append(0) # TODO number of requests for this app
            state.append(app.consumption_run)
            state.append(app.consumption_start)
            for pm in self.PMs:
                state.append(0) # TODO number of requests for this app on this PM

        return state


    def tick(self, action : list[float]) -> float:
        self.requests = self.traceGenerator.generate(self.apps)

        for pm in self.PMs:
            pm.resetLoad()

        for request in self.requests:
            pm_id = np.random.choice(range(len(self.PMs)), p=action[request])
            self.PMs[pm_id].addRequest(self.apps[request])
        
        energy_cost = 0
        for pm in self.PMs:
            energy_cost += pm.powerUsage()
        
        QoS = 0
        # TODO get QoS

        return energy_cost + LAMBDA*QoS
        
