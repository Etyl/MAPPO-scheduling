import json
from math import sin, pi
import numpy as np

import os
os.chdir(os.path.dirname(__file__))

apps = []

with open('../data/service.json', 'r') as file:
    data = json.load(file)
    for service in data:
        apps.append(service)

for app in apps:
    if app["distribution"][0] == "sin":
        app["distributionFunc"] = lambda x: int(app["distribution"][1]*sin(x/(2*pi*app["distribution"][3])+app["distribution"][4])+app["distribution"][2])

print("INIT SERVICE DATA")

def getRequests(time : int) -> list[int]:
    """
    Params:
        time : int : time of simulation
    Returns:
        requests : list[int] : array of size M corresponding to the number of requests per service
    """
    requests = np.zeros(len(apps))
    for app in apps:
        requests[app["id"]] = app["distributionFunc"](time)
    return requests