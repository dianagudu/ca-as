import numpy as np
import pandas as pd
import glob

from .stats import RawStats
from .stats import RawStatsOptimal
from .stats import RawStatsRandom
from .helper import Heuristic_Algorithm_Names
from .helper import Stochastic_Algorithm_Names


class RawStatsLoader():

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
        allstats = self.__load()
        # average over multiple runs when needed
        allstats = allstats.groupby(
            ['instance', 'algorithm']).mean().reset_index()
        # filter out non heuristic algos
        allstats = allstats[allstats.algorithm.isin(
                [x.name for x in Heuristic_Algorithm_Names])]
        return RawStats(self.name, allstats, [x.name for x in Heuristic_Algorithm_Names])

    def load_optimal(self):
        optstats = self.__load()
        # average over multiple runs when needed
        optstats = optstats.groupby(
            ['instance', 'algorithm']).mean().reset_index()
        return RawStatsOptimal(self.name, optstats)

    def load_random(self):
        randstats = self.__load()
        # filter out non-stochastic algos
        randstats = randstats[randstats.algorithm.isin(
                [x.name for x in Stochastic_Algorithm_Names])]
        return RawStatsRandom(self.name, randstats)

    def __load(self):
        allstats = pd.DataFrame()
        for stats_file in sorted(glob.glob(self.infolder + "/*")):
            stats = pd.read_csv(stats_file, header=None,
                                names=self.schema.keys(), dtype=self.schema)
            allstats = allstats.append(stats, ignore_index=True)
        return allstats
