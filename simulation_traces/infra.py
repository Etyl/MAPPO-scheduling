from new_types import App



class SBC:
    def __init__(self, id) -> None:
        self.id = 0
        self.load_CPU = 0
        self.load_IO = 0
        self.load_storage = 0
        self.currentApps = []
        self.usedApps = []
        self.startingApps = {}
        self.appRef : dict[App] = {}

    def powerUsage(self):
        if self.load_CPU == 0 and self.load_IO == 0:
            return 0
        return 2 + 2*self.load_CPU + 0.5*self.load_IO
    
    def resetLoad(self):
        self.load_CPU = 0
        self.load_IO = 0
        
        self.currentApps = self.usedApps.copy()
        self.usedApps = []

        removedApps = []
        for app in self.startingApps:
            self.startingApps[app] -= 1
            if self.startingApps[app] == 0:
                self.currentApps.append(app)
                removedApps.append(app)
            else:
                self.load_CPU += self.appRef[app].consumption_CPU_start
                self.load_IO += self.appRef[app].consumption_IO_start
        for app in removedApps:
            self.startingApps.pop(app)

    def addRequest(self, app: App):
        self.load_CPU += app.consumption_CPU
        self.load_IO += app.consumption_IO
        
        if app.id not in self.currentApps and app.id not in self.startingApps:
            self.startingApps[app.id] = 10
            self.appRef[app.id] = app
            return
        
        if app.id in self.startingApps:
            return

        if app.id not in self.usedApps:
            self.usedApps.append(app.id)

        self.load_CPU += app.consumption_CPU_start
        self.load_IO += app.consumption_IO_start
        self.load_storage += app.consumption_storage_start



class Cloud:
    def __init__(self) -> None:
        self.consumption_CPU = 0
        self.consumption_IO = 0
        self.load_CPU = 0
        self.load_IO = 0
        self.load_storage = 0
    
    def resetLoad(self):
        self.load_CPU = 0
        self.load_IO = 0

    def addRequest(self, app: App):
        self.load_CPU += app.consumption_CPU
        self.load_IO += app.consumption_IO
        self.load_storage += app.consumption_storage
