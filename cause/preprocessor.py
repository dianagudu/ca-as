import numpy as np
import pandas as pd
import glob

from .stats import RawStatsOptimal
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
        if isinstance(stats, RawStatsOptimal):
            stats = pd.DataFrame(
                self.stats.groupby('instance')
                          .apply(Preprocessor.__compute_costs_optimal))
        else:
            stats = pd.DataFrame(
                self.stats.groupby('instance')
                          .apply(Preprocessor.__compute_costs))

        costt = self.stats.pivot(
            index='instance', columns='algorithm', values='costt')
        costw = self.stats.pivot(
            index='instance', columns='algorithm', values='costw')

        return ProcessedStats(self.name,
                              stats.get_welfares(),
                              stats.get_times(),
                              costw[stats.algos],
                              costt[stats.algos])  # reorder columns by algo

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

    @staticmethod
    def __compute_costs_optimal(data):
        wcplex = data[data.algorithm == "CPLEX"].welfare.values[0]
        tcplex = data[data.algorithm == "CPLEX"].time.values[0]

        if wcplex == 0:
            data.eval('costw = 0', inplace=True)
        else:
            data.eval('costw = 1. - welfare / @wcplex', inplace=True)

        data.eval('costt = time / @tcplex', inplace=True)
        return data
