import numpy as np
import pandas as pd

from .helper import Heuristic_Algorithm_Names
from .plotter import Plotter

class Breakdown():

    def __init__(self, data, weights, algos, name):
        self.__data = data
        self.__weights = weights
        self.__algos = algos
        self.__name = name
        # todo validate input:
        # data is an np.array with dims (num algos, num weights)

    @property
    def data(self):
        return self.__data

    @property
    def weights(self):
        return self.__weights

    @property
    def algos(self):
        return self.__algos

    @property
    def name(self):
        return self.__name

    def save_to_latex(self, outfolder="/tmp", weight=1.):
        outfile = outfolder + "/breakdown_" + self.name
        index = np.where(self.weights==weight)[0][0]  # location for lambda=weight
        breakdown_perc = self.data[:,index] * 100. / self.data[:,index].sum()
        # write latex table to file
        with open(outfile, 'w') as f:
            for algo in range(self.data.shape[0]):
                f.write("&\t%s\t&\t%.2f\\%%\t\t\n" % (self.data[algo, index], breakdown_perc[algo]))

    def plot(self, outfolder="/tmp"):
        Plotter.plot_breakdown(self, outfolder)


class Postprocessor():

    def __init__(self, dataset):
        self.__dataset = dataset

    @property
    def dataset(self):
        return self.__dataset

    def breakdown(self):
        breakdown = np.empty(shape=(0,0))
        for weight in self.dataset.weights:
            column = self.dataset.lstats[weight].get_breakdown(self.dataset.algos)
            if breakdown.shape[0] == 0:
                breakdown = column
            else:
                breakdown = np.vstack([breakdown, column])
        breakdown = np.transpose(breakdown)

        return Breakdown(breakdown, self.dataset.weights,
                         self.dataset.algos, self.dataset.name)
