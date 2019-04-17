
from cause.predictor import Predictor


class PRAISEPredictor(Predictor):
    def __init__(self, lstats, features):
        super().__init__(lstats)

    def run(self, outfolder="/tmp"):
        pass