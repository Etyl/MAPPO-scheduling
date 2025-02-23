import json
from math import sin, pi
import numpy as np
import os

# Get the directory of the current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dname = os.path.dirname(dname)

apps = []

# Load service data from JSON file
with open(os.path.join(dname, 'data/service-poisson.json'), 'r') as file:
    data = json.load(file)
    for service in data:
        apps.append(service)

# Function to create a distribution function for each app
def create_distribution_func(distribution):
    if distribution[0] == "sin":
        A = distribution[1]
        B = distribution[2]
        C = distribution[3]
        D = distribution[4]
        return lambda x: int(A * sin(x / (2 * pi * C) + D) + B)
    
    elif distribution[0] == "poisson":
        A = distribution[1]
        return lambda x: np.random.poisson(A)
    
    else:
        raise ValueError("Unknown distribution type")

# Assign a unique distribution function to each app
for app in apps:
    app["distributionFunc"] = create_distribution_func(app["distribution"])

print("INIT SERVICE DATA")

def getRequests(time: int) -> np.ndarray[int]:
    """
    Params:
        time : int : time of simulation
    Returns:
        requests : np.ndarray[int] : array of size M corresponding to the number of requests per service
    """
    requests = np.zeros(len(apps))
    for app in apps:
        requests[app["id"]] = int(max(0, app["distributionFunc"](time)))
    return requests



# TODO poisson distribution