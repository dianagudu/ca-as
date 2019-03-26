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
            welfares.groupby(['instance', 'algorithm'])
                    .welfare.mean().reset_index(name='mean_welfare'))
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
                "[" + algo + "]", "min =",
                welfares[welfares.algorithm == algo].welfare.min(),
                ", max =",
                welfares[welfares.algorithm == algo].welfare.max())
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
            dobj = yaml.load(f, Loader=yaml.BaseLoader)
        return ProcessedStats.from_dict(dobj)

    def save(self, prefix):
        info = self.to_dict(prefix)
        with open(prefix + "_pstats.yaml", "w") as f:
            yaml.dump(info, f)
        self.welfares.to_csv(info["welfares"], float_format='%g')
        self.times.to_csv(info["times"], float_format='%g')
        self.costw.to_csv(info["costw"], float_format='%g')
        self.costt.to_csv(info["costt"], float_format='%g')

    @staticmethod
    def from_dict(dobj):
        welfares = pd.read_csv(dobj["welfares"], index_col='instance')
        times = pd.read_csv(dobj["times"], index_col='instance')
        costw = pd.read_csv(dobj["costw"], index_col='instance')
        costt = pd.read_csv(dobj["costt"], index_col='instance')
        return ProcessedStats(dobj["name"], dobj["algos"],
                              welfares, times, costw, costt)

    def to_dict(self, prefix):
        return {
            "name": self.name,
            "algos": self.algos,
            "welfares": prefix + ".welfares",
            "times": prefix + ".times",
            "costw": prefix + ".costw",
            "costt": prefix + ".costt"
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

    def get_breakdown(self, algos):
        elements, counts = np.unique(self.winners.winner, return_counts=True)

        # create column for weight and add to matrix
        column = [counts[np.where(elements == algo)[0]]
                  for algo in algos]
        column = [0 if column[i].size == 0 else column[i][0]
                  for i in range(0, len(column))]
        column = np.asarray(column)
        return column

    @staticmethod
    def load(filename, weight):
        costs = pd.read_csv(filename + ".costs", index_col='instance')
        winners = pd.read_csv(filename + ".winners", index_col='instance')
        return LambdaStats(weight, costs, winners)

    def save(self, filename):
        self.costs.to_csv(filename + ".costs", float_format='%g')
        self.winners.to_csv(filename + ".winners", float_format='%g')


class ProcessedDataset():

    def __init__(self, pstats, weights, lstats):
        self.__pstats = pstats
        self.__weights = weights
        self.__lstats = lstats

    @property
    def pstats(self):
        return self.__pstats

    @property
    def weights(self):
        return self.__weights

    @property
    def lstats(self):
        return self.__lstats

    @property
    def name(self):
        return self.__pstats.name

    @property
    def algos(self):
        return self.__pstats.algos

    @staticmethod
    def load(metafile):
        with open(metafile, "r") as f:
            dobj = yaml.load(f, Loader=yaml.BaseLoader)
        return ProcessedDataset.from_dict(dobj)

    @staticmethod
    def from_dict(dobj):
        pstats = ProcessedStats.load(dobj["pstats_file"])
        weights = np.array(dobj["weights"], dtype='float64')
        lstats_file_prefix = dobj["lstats_file_prefix"]
        lstats = {}
        for weight in weights:
            lstats[weight] = LambdaStats.load(
                lstats_file_prefix + str(weight), weight)
        return ProcessedDataset(pstats, weights, lstats)
