import numpy as np
import pandas as pd
import glob

from .helper import Algorithm_Names
from .helper import Stochastic_Algorithm_Names
from .helper import Heuristic_Algorithm_Names


class StatsLoader():
    __schema = {'instance': np.object_,
                'algorithm': np.object_,
                'time': np.float64,
                'welfare': np.float64,
                'ngoods': np.int64,
                'nwin': np.int64,
                'util_mean': np.float64,
                'util_stddev': np.float64,
                'price_mean': np.float64}

    def __init__(self, infolder, name):
        self.__infolder = infolder
        self.__name = name

    @property
    def infolder(self):
        return self.__infolder

    @property
    def schema(self):
        return self.__schema

    @property
    def name(self):
        return self.__name

    def load(self):
        allstats = pd.DataFrame()
        for stats_file in sorted(glob.glob(self.infolder + "/*")):
            stats = pd.read_csv(stats_file, header=None,
                                names=self.schema.keys(), dtype=self.schema)
            stats = stats.groupby(
                ['instance', 'algorithm']).mean().reset_index()
            allstats = allstats.append(stats, ignore_index=True)
        return AllStats(self.name, allstats)

    def load_random(self):
        randstats = pd.DataFrame()
        for stats_file in sorted(glob.glob(self.infolder + "/*")):
            stats = pd.read_csv(stats_file, header=None,
                                names=self.schema.keys(), dtype=self.schema)
            # filter out non-stochastic algos
            stats = stats[stats.algorithm.isin(
                          [x.name for x in Stochastic_Algorithm_Names]
                          )]
            randstats = randstats.append(stats, ignore_index=True)
        return RandStats(self.name, randstats)


class AllStats():

    def __init__(self, name, df):
        self.__name = name
        self.__df = df
        self.__algos = [x.name for x in Algorithm_Names]

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

    def get_welfares_feasible(self):
        welfares = self.get_welfares()
        # get infeasible instances and drop rows
        index_infeasible = welfares.index[welfares.CPLEX == 0]
        return welfares.drop(index_infeasible)

    def get_times(self):
        times = self.df.pivot(index='instance', columns='algorithm', values='time')
        return times[self.algos]

    def get_times_feasible(self):
        welfares = self.get_welfares()
        times = self.get_times()
        # get infeasible instances and drop rows
        index_infeasible = welfares.index[welfares.CPLEX == 0]
        return times.drop(index_infeasible)


class RandStats(AllStats):

    def __init__(self, name, df):
        super().__init__(name=name, df=df)
        self.__algos = [x.name for x in Stochastic_Algorithm_Names]

    @property
    def algos(self):
        return self.__algos


class HeuristicStats(AllStats):

    def __init__(self, name, df):
        super().__init__(name=name, df=df)
        self.__algos = [x.name for x in Heuristic_Algorithm_Names]