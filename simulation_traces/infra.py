from collections import deque

class App:
    def __init__(self, id, consumption_CPU = 0, consumption_BW = 0, consumption_run = 0, consumption_start = 0, distribution = lambda x: 0) -> None:
        self.id = id

        self.consumption_CPU = consumption_CPU # k instructions needed to execute a request
        self.consumption_BW = consumption_BW # kilo bytes needed to execute a request

        self.consumption_run = consumption_run # k instructions needed to run the service
        self.consumption_start = consumption_start # k instructions needed to start the service

        self.distribution = distribution

class PM:
    def __init__(self, id, n_apps, cpu = 0, bw = 0, consumption_idle = 0, consumption_max = 0) -> None:
        self.id = id

        self.CPU = cpu # max k instructions per second
        self.BW = bw # max kilo bytes per second

        self.CPU_load = 0 # current k instructions per second
        self.BW_load = 0 # current kilo bytes per second

        self.consumption_idle = consumption_idle # energy consumed when idle
        self.consumption_max = consumption_max # energy consumed when at max load

        self.n_apps = n_apps
        self.currentApps = [0]*n_apps
        self.lastApps = [0]*n_apps

    def powerUsage(self):
        if self.CPU_load == 0 and self.BW_load == 0:
            return 0
        
        energy = self.consumption_idle
        energy += (self.consumption_max-self.consumption_idle)*(0.8*(self.CPU_load/self.CPU) + 0.2*(self.BW_load/self.BW))

        for app,old_app in zip(self.currentApps, self.lastApps):
            if app != 0:
                energy += app.consumption_run
                if old_app != 0:
                    energy += app.consumption_start

        return energy
    
    def resetLoad(self):
        self.CPU_load = 0
        self.BW_load = 0
        
        self.usedApps = self.currentApps.copy()
        self.currentApps = [0]*self.n_apps

    def addRequest(self, app: App):
        self.currentApps[app.id] += 1

        self.CPU_load += app.consumption_CPU
        self.BW_load += app.consumption_BW
        

class SBC(PM):
    def __init__(self, id) -> None:
        super().__init__(id, 1000, 100, 2, 4)

class Cloud(PM):
    def __init__(self, id) -> None:
        super().__init__(id, 100000, 10000, 10, 1000)
