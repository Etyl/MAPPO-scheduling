from infra import App
import numpy as np


class TraceGenerator:

    appTypes = ["constant", "normal", "sinusoidal", "random"]

    def __init__(self):
        self.tick : int = 0
        self.requests : list[int] = []

    def addApp(self, app):
        if app is None:
            return
        self.apps.append(app)

    def generate(self, apps : list[App]) -> list[int]:
        self.requests = [0]*len(apps)
        for app in apps:
            self.requests[app.id] = app.distribution(self.tick)
        self.tick += 1

    def getRequest(self):
        s = sum(self.requests)
        if s <= 0:
            return None
        
        request = np.random.choice(range(len(self.requests)), p=[r/s for r in self.requests])
        self.requests[request] -= 1
        return request

