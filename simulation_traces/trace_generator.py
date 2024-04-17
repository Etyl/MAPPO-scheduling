from new_types import App
import numpy as np


class TraceGenerator:

    appTypes = ["constant", "normal", "sinusoidal", "random"]

    def __init__(self):
        self.tick : int = 0

    def addApp(self, app):
        if app is None:
            return
        self.apps.append(app)

    def generate(self, apps : list[App]):
        requests = []
        for app in apps:
            requests = requests + [app.id]*app.distribution(self.tick)
        self.tick += 1
        np.random.shuffle(requests)
        return requests

