from constants import *
import numpy as np

class Predictor:
    def __init__(self):
        pass

    def predict(self, requests):

        if len(requests) == 0:
            return [0]*N_APPS
        
        if len(requests) < 2:
            return requests[-1]

        return np.rint(requests[-1] + (requests[-1] - requests[-2])/TIME_PERIOD).astype(int)
