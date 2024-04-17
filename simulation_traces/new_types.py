

class Request():
    def __init__(self, appId):
        self.appId = appId

class App:
    def __init__(self) -> None:
        self.id = 0
        self.consumption_CPU = 0
        self.consumption_IO = 0

        self.consumption_CPU_start = 0
        self.consumption_IO_start = 0
        self.consumption_storage_start = 0

        self.distribution = lambda x: 0


class State:
    def __init__(self):
        self.nodeLoadCPU = []
        self.nodeLoadIO = []
        self.nodeLoadStorage = []
        self.nodeApps = []
