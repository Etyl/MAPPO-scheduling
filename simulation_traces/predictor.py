from constants import T


class Predictor:
    def __init__(self):
        pass

    def predict(self, requests):

        if len(requests) < 2:
            return requests[-1]

        return 2*requests[-1] - requests[-2]