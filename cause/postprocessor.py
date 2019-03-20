import numpy as np
import pandas as pd

from cause.helper import Heuristic_Algorithm_Names


class Breakdown():
    def __init__(self, data, weights, algos):
        self.__data = data
        self.__weights = weights
        self.__algos = algos
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

    def save_to_latex(self, outfile, weight=1.):
        index = np.where(self.weights==weight)[0][0]  # location for lambda=weight
        breakdown_perc = self.data[:,index] * 100. / self.data[:,index].sum()
        # write latex table to file
        with open(outfile, 'w') as f:
            for algo in range(self.data.shape[0]):
                f.write("&\t%s\t&\t%.2f\\%%\t\t\n" % (self.data[algo, index], breakdown_perc[algo]))


class Postprocessor():
    def __init__(self, stats):
        self.__stats = stats

    @property
    def stats(self):
        return self.__stats

    def get_breakdown(self, weights, outfolder="/tmp"):
        # extract welfares and times from stats
        welfares = self.stats.get_welfares_feasible()
        times = self.stats.get_times_feasible()

        # normalize to obtain cost of welfare and time for each algorithm
        cw = 1. - welfares.div(welfares.CPLEX, axis=0)
        ct = times.div(times.CPLEX, axis=0)
        # infeasible instance (cplex welfare is 0.) get cw = 0.
        #cw = cw.fillna(0.)

        breakdown = Postprocessor.__breakdown(cw, ct, weights)

        # save to file for latex table
        outfile = outfolder + "/breakdown_" + self.stats.name
        breakdown.save_to_latex(outfile)

        return breakdown

    @staticmethod
    def __breakdown(cw, ct, weights):
        halgos = [a.name for a in Heuristic_Algorithm_Names]
        cw = cw[halgos]
        ct = ct[halgos]
        breakdown = np.empty(shape=(0,0))
        for weight in weights:
            # compute cost for weight
            costs = ((weight**2 * cw.pow(2)).add((1-weight)**2 * ct.pow(2))).pow(0.5)

            # get winners for weight
            y = costs.apply(lambda x: x.idxmin(), axis=1).values
            y = np.array([halgos.index(algo) for algo in y])

            # get breakdown by class
            elements, counts = np.unique(y, return_counts=True)

            # create column for weight and add to matrix
            column = [counts[np.where(elements == algo)[0]] for algo in range(len(halgos))]
            column = [0 if column[i].size == 0 else column[i][0] for i in range(0, len(column))]
            column = np.asarray(column)
            if breakdown.shape[0] == 0:
                breakdown = column
            else:
                breakdown = np.vstack([breakdown, column])
        breakdown = np.transpose(breakdown)

        return Breakdown(breakdown, weights, halgos)