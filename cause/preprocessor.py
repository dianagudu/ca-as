import numpy as np
import pandas as pd
import glob

from .stats import ProcessedStats

class Preprocessor():

    def __init__(self, name, stats):
        self.__name = name
        self.__stats = stats

    @property
    def name(self):
        return self.__name

    @property
    def stats(self):
        return self.__stats

    def process(self):
        stats = pd.DataFrame(
            self.stats.groupby('instance').apply(Preprocessor.__compute_costs))
        return ProcessedStats(self.name, stats)

    @staticmethod
    def __compute_costs(data):
        wmin = data.welfare.min()
        wmax = data.welfare.max()
        tmin = data.time.min()
        tmax = data.time.max()
        if wmax - wmin == 0:
            data.eval('costw = 0', inplace=True)
        else:
            data.eval(
                'costw = (@wmax - welfare) / (@wmax - @wmin)', inplace=True)
        if tmax - tmin == 0:
            data.eval('costt = 0', inplace=True)
        else:
            data.eval('costt = (time - @tmin) / (@tmax - @tmin)', inplace=True)
        return data
