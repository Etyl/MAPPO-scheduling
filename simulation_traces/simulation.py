from trace_generator import TraceGenerator
from new_types import App, State
import random
from math import *
from infra import SBC
from scheduler import Scheduler
import numpy as np
import matplotlib.pyplot as plt

SCALE = 20

# TODO
# Spectrum of IO -> CPU
# collect data for precise simulation
# scalable input
# output / input ?
# reward min energy / max QoS


class Controller:
    def __init__(self) -> None:
        self.traceGenerator : TraceGenerator = TraceGenerator()
        self.scheduler : Scheduler = Scheduler()
        self.requests : list[int] = []
        self.nodes = [SBC(i) for i in range(4)]
        self.apps : list[App] = []
    
        serverApp = App()
        serverApp.id = 0
        serverApp.consumption_CPU = 0.002
        serverApp.consumption_CPU_start = 0.05
        serverApp.distribution = lambda x: int(SCALE*sin(x/(2*pi*10))+1.2*SCALE)
        self.apps.append(serverApp)

        serverApp = App()
        serverApp.id = 1
        serverApp.consumption_CPU = 0.002
        serverApp.consumption_CPU_start = 0.05
        serverApp.distribution = lambda x: int(0.6*SCALE*sin(x/(2*pi*10)+140)+SCALE)
        self.apps.append(serverApp)

        serverApp = App()
        serverApp.id = 2
        serverApp.consumption_CPU = 0.002
        serverApp.consumption_CPU_start = 0.05
        serverApp.distribution = lambda x: int(0.3*SCALE*sin(x/(2*pi*10)+180)+0.3*SCALE)
        self.apps.append(serverApp)


    def getState(self, request : int = -1):
        state = State()
        state.nodeLoadCPU = [node.load_CPU for node in self.nodes]
        state.nodeLoadIO = [node.load_IO for node in self.nodes]
        state.nodeLoadStorage = [node.load_storage for node in self.nodes]
        state.nodeApps = [ int(request in node.currentApps) for node in self.nodes]
        return state



    def tick(self) -> None:
        self.requests = self.traceGenerator.generate(self.apps)

        for node in self.nodes:
            node.resetLoad()

        for request in self.requests:
            schedule = self.scheduler.assign(request, self.getState(request))
            i = np.argmax(schedule)
            if i == len(schedule)-1:
                # TODO : send to cloud
                continue
            self.nodes[i].addRequest(self.apps[request])


def main():
    cpu_loads = []
    requests_total = []
    energy_consumption = []
    apps_per_nodes = []
    controller = Controller()
    for i in range(500):
        controller.tick()
        cpu_loads.append([node.load_CPU for node in controller.nodes])
        requests_total.append(controller.requests)
        energy_consumption.append([node.powerUsage() for node in controller.nodes])
        apps_per_nodes.append([len(node.currentApps) for node in controller.nodes])
    
    plt.figure()
    for i in range(len(controller.nodes)):
        plt.plot([cpu_loads[j][i] for j in range(len(cpu_loads))])
    plt.title("CPU Load of each node")
    
    plt.figure()
    for app in controller.apps:
        plt.plot([np.sum([x == app.id for x in requests_total[i]]) for i in range(len(requests_total))])
    plt.title("Requests for each app")

    plt.figure()
    for i in range(len(controller.nodes)):
        plt.plot([energy_consumption[j][i] for j in range(len(energy_consumption))])
    plt.plot([np.sum([energy_consumption[j][i] for i in range(len(energy_consumption[0]))]) for j in range(len(energy_consumption))])
    plt.title("Power consumption")

    plt.figure()
    energy_consumption = np.array(energy_consumption)
    plt.plot([np.sum([energy_consumption[:j+1,i] for i in range(len(energy_consumption[0]))]) for j in range(len(energy_consumption))])
    plt.title("Total power consumption")

    plt.figure()
    for i in range(len(controller.nodes)):
        plt.plot([apps_per_nodes[j][i] for j in range(len(apps_per_nodes))])
    plt.title("Apps per node")


    
    plt.show()


if __name__ == "__main__":
    main()
    