import pandas as pd

from .helper import Algorithm_Names
from .helper import Stochastic_Algorithm_Names

class RawStats():
    def __init__(self, name, df, algos):
        self.__name = name
        self.__df = df
        self.__algos = algos

    @property
    def df(self):
        return self.__df

    @property
    def name(self):
        return self.__name

    @property
    def data(self):
        return self.__df.values

    @property
    def columns(self):
        return self.__df.columns

    @property
    def algos(self):
        return self.__algos

    def save(self, filename):
        pass

    def get_welfares(self):
        welfares = self.df.pivot(index='instance', columns='algorithm', values='welfare')
        # reorders columns
        return welfares[self.algos]

    def get_times(self):
        times = self.df.pivot(index='instance', columns='algorithm', values='time')
        return times[self.algos]


class RawStatsOptimal(RawStats):
    def __init__(self, name, df):
        super().__init__(name, df, [x.name for x in Algorithm_Names])

    def get_welfares_feasible(self):
        welfares = self.get_welfares()
        # get infeasible instances and drop rows
        index_infeasible = welfares.index[welfares.CPLEX == 0]
        return welfares.drop(index_infeasible)

    def get_times_feasible(self):
        welfares = self.get_welfares()
        times = self.get_times()
        # get infeasible instances and drop rows
        index_infeasible = welfares.index[welfares.CPLEX == 0]
        return times.drop(index_infeasible)


class RawStatsRandom(RawStats):
    def __init__(self, name, df):
        super().__init__(name, df, [x.name for x in Stochastic_Algorithm_Names])

    def get_welfares(self):
        return self.df[['instance', 'algorithm', 'welfare']]

    def get_times(self):
        return self.df[['instance', 'algorithm', 'time']]

    def get_normalized_welfares(self):
        welfares = self.get_welfares()

        # normalize welfare by average value on each instance
        welfares_means = pd.DataFrame(
            welfares.groupby(['instance', 'algorithm']).welfare.mean().reset_index(name='mean_welfare'))
        welfares = welfares.merge(welfares_means)
        welfares.eval(
            'welfare = (welfare / mean_welfare - 1.) * 100.', inplace=True)
        welfares = welfares.dropna()
        return welfares


class ProcessedStats():
    def __init__(self, name, df):
        self.__name = name
        self.__df = df
        # df columns: instance, algorithm, welfares, times, costw, costt

    @property
    def name(self):
        return self.__name

    @property
    def df(self):
        return self.__df