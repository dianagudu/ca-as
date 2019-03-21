import numpy as np
import pandas as pd
import glob

from .stats import RawStatsOptimal
from .stats import ProcessedStats
from .stats import LambdaStats


class StatsPreprocessor():

    def __init__(self, stats):
        self.__stats = stats

    @property
    def stats(self):
        return self.__stats

    def process(self):
        if isinstance(self.stats.df, RawStatsOptimal):
            pstats = pd.DataFrame(
                self.stats.df.groupby('instance')
                          .apply(StatsPreprocessor.__compute_costs_optimal))
        else:
            pstats = pd.DataFrame(
                self.stats.df.groupby('instance')
                          .apply(StatsPreprocessor.__compute_costs))

        costt = pstats.pivot(
            index='instance', columns='algorithm', values='costt')
        costw = pstats.pivot(
            index='instance', columns='algorithm', values='costw')

        return ProcessedStats(self.stats.name,
                              self.stats.get_welfares(),
                              self.stats.get_times(),
                              costw[self.stats.algos],
                              costt[self.stats.algos])  # reorder columns by algo

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


class LambdaStatsPreprocessor():

    def __init__(self, pstats):
        self.__pstats = pstats

    @property
    def pstats(self):
        return self.__pstats

    def process(self, weight):
        costs = ((weight * self.pstats.costw) ** 2 +
                ((1 - weight) * self.pstats.costt) ** 2) ** 0.5
        winners = costs.idxmin(axis=1)
        return LambdaStats(weight, costs, winners)

