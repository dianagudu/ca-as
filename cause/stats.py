import numpy as np
import pandas as pd
import yaml

from .helper import Algorithm_Names
from .helper import Stochastic_Algorithm_Names
from .plotter import Plotter


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

    def get_welfares(self):
        welfares = self.df.pivot(
            index='instance', columns='algorithm', values='welfare')
        # reorders columns
        return welfares[self.algos]

    def get_times(self):
        times = self.df.pivot(
            index='instance', columns='algorithm', values='time')
        return times[self.algos]

    def save(self, filename):
        pass

    @staticmethod
    def load(filename):
        pass


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

    def plot(self, outfolder="/tmp"):
        outfile_welfare = outfolder + "/" + "welfare_" + self.name
        outfile_time = outfolder + "/" + "time_" + self.name

        welfares = self.get_welfares_feasible()
        times = self.get_times_feasible()

        # normalize welfare and time by values of optimal algorithm (cplex)
        welfares = welfares.div(welfares.CPLEX, axis=0).multiply(100., axis=0)
        times = times.div(times.CPLEX, axis=0).multiply(100., axis=0)

        Plotter.boxplot_average_case(
            welfares.values, self.algos, outfile_welfare,
            ylabel="% of optimal welfare (CPLEX)")
        Plotter.boxplot_average_case(
            times.values, self.algos, outfile_time,
            top=100000, bottom=0.01, ylog=True,
            ylabel="% of time of optimal algorithm (CPLEX)")


class RawStatsRandom(RawStats):

    def __init__(self, name, df):
        super().__init__(
            name, df, [x.name for x in Stochastic_Algorithm_Names])

    def get_welfares(self):
        return self.df[['instance', 'algorithm', 'welfare']]

    def get_times(self):
        return self.df[['instance', 'algorithm', 'time']]

    def __get_normalized_welfares(self):
        welfares = self.get_welfares()
        # normalize welfare by average value on each instance
        welfares_means = pd.DataFrame(
            welfares.groupby(['instance', 'algorithm']).welfare.mean().reset_index(name='mean_welfare'))
        welfares = welfares.merge(welfares_means)
        welfares.eval(
            'welfare = (welfare / mean_welfare - 1.) * 100.', inplace=True)
        welfares = welfares.dropna()
        return welfares

    def plot(self, outfolder="/tmp"):
        outfile = outfolder + "/random_" + self.name
        welfares = self.__get_normalized_welfares()
        data = []
        for algo in self.algos:
            data.append(welfares[welfares.algorithm == algo].welfare.values)
            print(
                "[" + algo + "]", "min =", welfares[
                    welfares.algorithm == algo].welfare.min(),
                ", max =", welfares[welfares.algorithm == algo].welfare.max())
        Plotter.boxplot_random(data, self.algos, outfile)


class ProcessedStats():

    def __init__(self, name, algos, welfares, times, costw, costt):
        self.__name = name
        self.__algos = algos
        self.__welfares = welfares
        self.__times = times
        self.__costw = costw
        self.__costt = costt

    @property
    def name(self):
        return self.__name

    @property
    def welfares(self):
        return self.__welfares

    @property
    def times(self):
        return self.__times

    @property
    def costw(self):
        return self.__costw

    @property
    def costt(self):
        return self.__costt

    @property
    def algos(self):
        return self.__algos

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            dobj = yaml.load(f)
        return ProcessedStats.from_dict(dobj)

    def save(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_dict(dobj):
        return ProcessedStats(**dobj)

    def to_dict(self):
        return {
            "name": self.name,
            "algos": self.algos,
            "welfares": self.welfares,
            "times": self.times,
            "costw": self.costw,
            "costt": self.costt
        }


class LambdaStats():

    def __init__(self, weight, costs, winners):
        self.__weight = weight
        self.__costs = costs
        self.__winners = winners

    @property
    def weight(self):
        return self.__weight

    @property
    def costs(self):
        return self.__costs

    @property
    def winners(self):
        return self.__winners

    # todo: fix this!
    def get_breakdown(self, algos):
        elements, counts = np.unique(self.winners, return_counts=True)

        # create column for weight and add to matrix
        column = [counts[np.where(elements == algo)[0]]
                  for algo in algos]
        column = [0 if column[i].size == 0 else column[i][0]
                  for i in range(0, len(column))]
        column = np.asarray(column)
        return column

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            dobj = yaml.load(f)
        return LambdaStats.from_dict(dobj)

    def save(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_dict(dobj):
        return LambdaStats(**dobj)

    def to_dict(self):
        return {
            "weight": self.weight,
            "costs": self.costs,
            "winners": self.winners
        }

class ProcessedDataset():
    def __init__(self, pstats, weights, l_lstats):
        self.__pstats = pstats
        self.__weights = weights
        self.__l_lstats = l_lstats

    @property
    def pstats(self):
        return self.__pstats

    @property
    def weights(self):
        return self.__weights

    @property
    def l_lstats(self):
        return self.__l_lstats

    @staticmethod
    def load(metafile):
        with open(metafile, "r") as f:
            dobj = yaml.load(f)
        return ProcessedDataset.from_dict(dobj)

    @staticmethod
    def from_dict(dobj):
        pstats = ProcessedStats.load(dobj["pstats_file"])
        weights = np.array(dobj["weights"])
        l_lstats = []
        for weight in weights:
            lstats = LambdaStats.load(dobj["lstats_files"][weight])
            l_lstats.append(lstats)
        return ProcessedDataset(pstats, weights, l_lstats)
